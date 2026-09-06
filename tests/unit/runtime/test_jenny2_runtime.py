from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import angler.runtime.jenny2_runtime as jenny2_runtime_module

from angler.runtime.higher_level_experience_cycle import (
    ConsequenceVector,
    ExecutionReceipt,
    MemoryCandidate,
    StructuredExperience,
)
from angler.runtime.higher_level_autonomy_adapter import (
    AdaptiveHumanTurnRouter,
    LearnedAffordanceSelection,
)
from angler.runtime.authored_artifact import (
    AUTHORED_ARTIFACT_AFFORDANCE,
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AUTHORED_ARTIFACT_SOURCE_KIND,
    AUTHORED_ARTIFACT_SOURCE_REF,
    AuthoredArtifactExecutor,
)
from angler.runtime.jenny2_runtime import (
    BoundedWebPageReader,
    FederatedWebResearchExecutor,
    InternalAffordanceBinding,
    NativeHostToolResult,
    QWEN38_BASE_SERVED_MODEL,
    QWEN38_LORA_TARGET_MODULES,
    ReadOnlyExternalAffordanceBinding,
    SGLangLoRABinding,
    assemble_jenny2_runtime,
)
from angler.runtime.jenny_library import (
    JennyLibraryExecutor,
    LIBRARY_AFFORDANCE,
    LIBRARY_AFFORDANCE_ID,
    LIBRARY_CATALOG_CONTRACT,
    LIBRARY_SOURCE_KIND,
)
from angler.runtime.jenny_composition import (
    ModelArtifactSnapshot,
    ModelBindingSnapshot,
)
from angler.runtime.jenny2_tool_bridge import (
    NativeModelToolCall,
    NativeOpenClawCatalog,
    NativeToolModelResult,
)
from angler.runtime.persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    CycleObservation,
    ObservableConsequence,
    RankedAffordance,
)
from angler.runtime.temporal_v2 import TrustedClock


QUALIFICATION = "sha256:" + "6" * 64


class _Source:
    def __init__(self):
        self.wall = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
        self.mono = 40_000_000_000

    def wall_now(self):
        value = self.wall
        self.wall += timedelta(milliseconds=1)
        return value

    def mono_now(self):
        value = self.mono
        self.mono += 1_000_000
        return value


class _Memory:
    def __init__(self):
        self.records = []

    def recall(self, request, *, limit):
        return (
            MemoryCandidate("sha256:" + "a" * 64, "Prior verified process.", 0.1),
        )[:limit]

    def store(self, content, provenance_refs):
        self.records.append((content, tuple(provenance_refs)))
        return "sha256:" + "b" * 64


class _Experience:
    def generate_experience(self, request, memories, temporal):
        return StructuredExperience(
            "Interpret constraints.",
            "VERIFY",
            "Try and verify.",
            "Observable evidence will change state.",
            ("Check receipt.",),
            0.3,
        )

    def reflect(self, request, receipt, consequence, experience):
        return "Observed consequence changed the process estimate."

    def consolidate(self, request, experience, reflection, consequence):
        return "Proposal: retain the verified consequence relation."


class _LibraryExperience(_Experience):
    def __init__(self):
        self.outcome_assessment_calls = 0

    def assess_observed_outcome(self, *args, **kwargs):
        del args, kwargs
        self.outcome_assessment_calls += 1
        raise AssertionError(
            "a source-bound library observation must not be reinterpreted "
            "before its public continuation"
        )


class _Cortex:
    def __init__(self):
        self.calls = 0

    def execute(self, request, experience, memories):
        self.calls += 1
        return ExecutionReceipt("sha256:" + "c" * 64, "Synthetic response.", "COMPLETED")


class _BlockingCortex:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def execute(self, request, experience, memories):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test cortex was not released")
        return ExecutionReceipt("sha256:" + "d" * 64, "Synthetic response.", "COMPLETED")


class _NativeController:
    def __init__(self):
        self.visible_by_source = {}
        self.update_calls = 0

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del memories, experience, state, capability_modules
        visible = tuple(item.affordance_id for item in affordances)
        self.visible_by_source.setdefault(observation.source, []).append(visible)
        selected = (
            "openclaw.supervised.turn"
            if observation.source == "AGENT"
            else "cortex.respond"
        )
        rankings = tuple(
            RankedAffordance(item.affordance_id, 1.0 if item.affordance_id == selected else 0.0)
            for item in affordances
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected,
            rankings=rankings,
            action_payload=observation.content or "Inspect one learned pending concern.",
            state_assessment="The owner request may require one guarded host observation.",
            resolution_target="Resolve the request from source-bound evidence.",
            expected_state_delta="The requested evidence becomes available for a final answer.",
            evidence_refs=(temporal.sample_ref,),
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        self.update_calls += 1
        return parent_state


class _FullLibraryRouterBackend:
    model_ref = "sha256:" + "e" * 64

    def __init__(self):
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user, max_new_tokens))
        affordances = [
            item["affordance_id"] for item in json.loads(user)["affordances"]
        ]
        affordances.sort(key=lambda item: item != LIBRARY_AFFORDANCE_ID)
        return json.dumps(
            {
                "route": "FULL_DELIBERATION",
                "response": "",
                "interpretation": "The library catalog has not yet been observed.",
                "rationale": "A read-only observation may resolve the human request.",
                "predicted_consequence": "The controller compares an observation with speech.",
                "uncertainty": 0.35,
                "reasons_for_direct_response": [],
                "reasons_for_further_deliberation": [
                    "Null catalog state does not establish an empty catalog."
                ],
                "unfinished_patterns": [],
                "next_internal_request": "",
                "affordance_preference_order": affordances,
                "required_capability_keys": [],
                "evidence_keys": ["observation"],
            }
        )


class _HumanLibraryController:
    def __init__(self):
        self.calls = []

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del memories, experience, state, capability_modules
        visible = tuple(item.affordance_id for item in affordances)
        self.calls.append((observation, visible))
        selected = (
            LIBRARY_AFFORDANCE_ID
            if observation.source == "HUMAN"
            else "cortex.respond"
        )
        action_payload = (
            json.dumps(
                {
                    "operation": "list",
                    "reading_purpose": (
                        "Observe the exact eligible catalog before answering."
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            if selected == LIBRARY_AFFORDANCE_ID
            else observation.content
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected,
            rankings=tuple(
                RankedAffordance(
                    item.affordance_id,
                    1.0 if item.affordance_id == selected else 0.0,
                )
                for item in affordances
            ),
            action_payload=action_payload,
            state_assessment="The current request has one bounded evidence gap.",
            resolution_target="Answer from an exact source-bound catalog observation.",
            expected_state_delta="The human receives an evidence-grounded public answer.",
            evidence_refs=(temporal.sample_ref,),
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        return parent_state


class _LibraryGroundedCortex:
    def __init__(self):
        self.calls = []

    def execute_with_context(
        self, request, experience, memories, temporal_now, cognitive_state
    ):
        self.calls.append(
            (request, experience, tuple(memories), temporal_now, cognitive_state)
        )
        observation = cognitive_state["library_turn_observation"][
            "observable_consequence"
        ]["observation"]
        return ExecutionReceipt(
            "sha256:" + "f" * 64,
            f"I can see {len(observation['eligible_items'])} eligible works in the verified catalog.",
            "COMPLETED",
        )


class _NativeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return NativeToolModelResult(
                route="TOOL_REQUESTED",
                final_content=None,
                tool_calls=(
                    NativeModelToolCall(
                        "call-native-1",
                        "web.search",
                        '{"query":"moving origin"}',
                    ),
                ),
                choice_json='{"decision":"request_source_bound_search"}',
                finish_reason="tool_calls",
            )
        if len(self.calls) == 2:
            return NativeToolModelResult(
                route="FINAL",
                final_content="The host-observed result is forty-two.",
                tool_calls=(),
                choice_json='{"decision":"answer_from_observed_result"}',
                finish_reason="stop",
            )
        raise AssertionError("committed native turn unexpectedly called the model again")


class _TwoToolNativeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return NativeToolModelResult(
                route="TOOL_REQUESTED",
                final_content=None,
                tool_calls=(
                    NativeModelToolCall(
                        "call-search-1",
                        "web.search",
                        '{"query":"moving origin"}',
                    ),
                    NativeModelToolCall(
                        "call-fetch-1",
                        "web.fetch",
                        '{"url":"https://example.test/evidence"}',
                    ),
                ),
                choice_json='{"decision":"request_two_source_bound_observations"}',
                finish_reason="tool_calls",
            )
        if len(self.calls) == 2:
            return NativeToolModelResult(
                route="FINAL",
                final_content="Both host-observed results support the answer.",
                tool_calls=(),
                choice_json='{"decision":"answer_from_two_observed_results"}',
                finish_reason="stop",
            )
        raise AssertionError("native replay unexpectedly called the model again")


def _native_tools():
    return (
        {
            "name": "web.search",
            "description": "Search for one owner-requested public fact.",
            "input_schema": {
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "type": "object",
            },
            "external_effect": False,
        },
    )


def _native_two_tools():
    return (
        *_native_tools(),
        {
            "name": "web.fetch",
            "description": "Fetch one exact owner-requested public resource.",
            "input_schema": {
                "additionalProperties": False,
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "type": "object",
            },
            "external_effect": False,
        },
    )


def _evaluate(receipt):
    return ConsequenceVector(0.7, 0.8, 0.2, 0.5, 0.8, 0.6, 0.1, 1.0)


class Jenny2RuntimeTests(unittest.TestCase):
    def test_live_composition_is_fresh_transient_model_context_only(self):
        class CompositionAwareExperience(_Experience):
            def __init__(self):
                self.cognitive_states = []

            def generate_situated_experience(
                self, request, memories, temporal, *, cognitive_state
            ):
                self.cognitive_states.append(cognitive_state)
                return super().generate_experience(request, memories, temporal)

        model = ModelBindingSnapshot(
            kind="BASE_CONTROL",
            binding_ref=None,
            binding_file=None,
            base_served_model="test-cortex",
            configured_served_model="test-cortex",
            observed_served_model="test-cortex",
            runtime_ref="sha256:" + "1" * 64,
            runtime_image="test-runtime@sha256:" + "2" * 64,
            runtime_revision="test-revision",
            source_model=ModelArtifactSnapshot(
                path="/models/source",
                revision="source-revision",
                config_sha256="3" * 64,
                index_sha256="4" * 64,
            ),
            quantized_model=ModelArtifactSnapshot(
                path="/models/quantized",
                revision="quantized-revision",
                config_sha256="5" * 64,
                index_sha256="6" * 64,
            ),
            adapter=None,
            selector_qualification_ref=QUALIFICATION,
        )
        experience = CompositionAwareExperience()
        with tempfile.TemporaryDirectory() as directory:
            runtime = assemble_jenny2_runtime(
                Path(directory) / "composition.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=experience,
                cortex=_Cortex(),
                selector_qualification_ref=QUALIFICATION,
                composition_model_binding=model,
                composition_evidence_refs=(QUALIFICATION,),
            )
            before_state = runtime.supervisor.state_bytes()
            manifest = runtime.composition_manifest()
            overlay = runtime.composition_overlay()

            self.assertEqual(manifest.integrity_status, "FRESH")
            self.assertTrue(overlay["SELF"]["system"]["claims_current"])
            self.assertEqual(
                overlay["SELF"]["system"]["cortex"]["served_model"],
                "test-cortex",
            )
            self.assertEqual(runtime.supervisor.state_bytes(), before_state)
            self.assertEqual(runtime.supervisor.episode_items(), ())

            result = runtime.chat("composition:human", "What system are you using?")
            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(len(experience.cognitive_states), 1)
            supplied = experience.cognitive_states[0]["trusted_composition"]
            self.assertEqual(
                supplied["SELF"]["system"]["cortex"]["served_model"],
                "test-cortex",
            )
            episode = runtime.supervisor.episode_item(result.episode_ref)
            self.assertNotIn("trusted_composition", episode.payload_json)
            self.assertNotIn(
                "trusted_composition",
                runtime.supervisor.state_bytes().decode("utf-8"),
            )
            refreshed = runtime.composition_manifest()
            self.assertEqual(refreshed.integrity_status, "FRESH")
            self.assertNotEqual(refreshed.manifest_ref, manifest.manifest_ref)

    def test_sglang_lora_binding_is_content_addressed_and_fail_closed(self):
        def write_json(path, value):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            quantized = root / "quantized"
            run = root / "run"
            adapter = run / "adapter"
            source.mkdir()
            quantized.mkdir()
            adapter.mkdir(parents=True)
            write_json(source / "config.json", {"model": "source"})
            write_json(source / "model.safetensors.index.json", {"weight_map": {}})
            write_json(quantized / "config.json", {"model": "quantized"})
            write_json(
                quantized / "model.safetensors.index.json", {"weight_map": {}}
            )
            write_json(
                quantized / "conversion-manifest.json",
                {
                    "source": {
                        "repository": "Qwen/Qwen3.8-27B",
                        "revision": jenny2_runtime_module.QWEN38_SOURCE_REVISION,
                    }
                },
            )
            write_json(
                quantized / "qualification.json",
                {
                    "checkpoint": {
                        "source_repository": "Qwen/Qwen3.8-27B",
                        "source_revision": jenny2_runtime_module.QWEN38_SOURCE_REVISION,
                    }
                },
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
            adapter_config = {
                "base_model_name_or_path": str(source),
                "lora_alpha": 16,
                "peft_type": "LORA",
                "r": 8,
                "target_modules": list(reversed(QWEN38_LORA_TARGET_MODULES)),
                "task_type": "CAUSAL_LM",
                "use_rslora": False,
            }
            write_json(adapter / "adapter_config.json", adapter_config)

            patches = (
                patch.object(jenny2_runtime_module, "QWEN38_SOURCE_MODEL_PATH", source),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_SOURCE_CONFIG_SHA256",
                    sha(source / "config.json"),
                ),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_SOURCE_INDEX_SHA256",
                    sha(source / "model.safetensors.index.json"),
                ),
                patch.object(
                    jenny2_runtime_module, "QWEN38_NVFP4_MODEL_PATH", quantized
                ),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_NVFP4_CONFIG_SHA256",
                    sha(quantized / "config.json"),
                ),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_NVFP4_INDEX_SHA256",
                    sha(quantized / "model.safetensors.index.json"),
                ),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_NVFP4_CONVERSION_SHA256",
                    sha(quantized / "conversion-manifest.json"),
                ),
                patch.object(
                    jenny2_runtime_module,
                    "QWEN38_NVFP4_QUALIFICATION_SHA256",
                    sha(quantized / "qualification.json"),
                ),
            )
            for active_patch in patches:
                active_patch.start()
            self.addCleanup(lambda: [active_patch.stop() for active_patch in patches])

            training_result = {
                "adapter_config_sha256": sha(adapter / "adapter_config.json"),
                "adapter_model_sha256": sha(adapter / "adapter_model.safetensors"),
                "base_config_sha256": sha(source / "config.json"),
                "base_index_sha256": sha(source / "model.safetensors.index.json"),
                "base_revision": jenny2_runtime_module.QWEN38_SOURCE_REVISION,
                "curriculum_manifest_sha256": "1" * 64,
                "evaluation_sha256": "2" * 64,
                "held_out_teacher_forced_evaluation": {"examples": 1},
                "lora_scaling": "standard-alpha-over-r",
                "max_steps": 2,
                "rank": 8,
                "schema": "jenny2.qlora-training.v1",
                "status": "PASS",
                "steps_completed": 2,
                "target_modules": list(QWEN38_LORA_TARGET_MODULES),
                "text_only_training": True,
                "training_profile": "mixed-initial-v1",
                "training_sha256": "3" * 64,
            }
            write_json(run / "training-result.json", training_result)
            binding = SGLangLoRABinding.from_training_run(run)
            binding_path = root / "binding.json"
            write_json(binding_path, binding.to_payload())
            restored = SGLangLoRABinding.from_file(binding_path)

            self.assertEqual(restored, binding)
            self.assertTrue(
                binding.served_model.startswith(
                    QWEN38_BASE_SERVED_MODEL + "--lora-sha256-"
                )
            )
            argv = binding.docker_argv(cache_path=root / "cache")
            self.assertIn("SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1", argv)
            self.assertIn("--lora-strict-loading", argv)
            self.assertEqual(argv[argv.index("--lora-backend") + 1], "triton")
            self.assertEqual(argv[argv.index("--max-loaded-loras") + 1], "2")
            self.assertEqual(argv[argv.index("--max-loras-per-batch") + 1], "2")
            self.assertIn(f"type=bind,src={adapter},dst=/adapter,readonly", argv)
            self.assertNotIn("jenny-qwen38", argv)

            (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "adapter weights changed"):
                SGLangLoRABinding.from_file(binding_path)

    @staticmethod
    def _web_request(affordance_id, payload):
        return AffordanceRequest(
            "sha256:" + "1" * 64,
            "test.web.read",
            affordance_id,
            "sha256:" + "2" * 64,
            "sha256:" + "3" * 64,
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )

    def test_bounded_web_reader_rejects_private_targets_before_request(self):
        reader = BoundedWebPageReader(timeout_seconds=1)
        with self.assertRaisesRegex(ValueError, "public addresses"):
            reader(
                self._web_request(
                    "tool.web_read", {"url": "https://127.0.0.1/private"}
                )
            )

    def test_bounded_web_reader_returns_source_bound_sanitized_text(self):
        class Headers:
            @staticmethod
            def get_content_type():
                return "text/html"

            @staticmethod
            def get_content_charset():
                return "utf-8"

        class Response:
            headers = Headers()

            @staticmethod
            def geturl():
                return "https://example.org/research"

            @staticmethod
            def read(_limit):
                return (
                    b"<html><title>Evidence page</title><script>ignore me</script>"
                    b"<body>Observed finding.</body></html>"
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Opener:
            @staticmethod
            def open(_request, *, timeout):
                self.assertEqual(timeout, 1.0)
                return Response()

        resolved = [
            (
                2,
                1,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ]
        with patch(
            "angler.runtime.jenny2_runtime.socket.getaddrinfo",
            return_value=resolved,
        ), patch(
            "angler.runtime.jenny2_runtime.urllib.request.build_opener",
            return_value=Opener(),
        ):
            receipt = BoundedWebPageReader(timeout_seconds=1)(
                self._web_request(
                    "tool.web_read", {"url": "https://example.org/research"}
                )
            )
        self.assertEqual(receipt.status, "COMPLETED")
        observation = receipt.observable_consequence
        self.assertIsNotNone(observation)
        payload = json.loads(observation.observation_json)
        self.assertEqual(payload["title"], "Evidence page")
        self.assertIn("Observed finding.", payload["text"])
        self.assertNotIn("ignore me", payload["text"])
        self.assertEqual(observation.source_ref, BoundedWebPageReader.SOURCE_REF)
        self.assertEqual(observation.artifact_refs, observation.evidence_refs)

    def test_federated_web_research_deduplicates_evidence_references(self):
        executor = FederatedWebResearchExecutor(timeout_seconds=1)
        duplicate = {
            "provider": "test",
            "title": "Same result",
            "url": "https://example.org/same",
            "summary": "Same summary",
            "published": None,
        }
        executor._general = lambda _query, _limit: [dict(duplicate), dict(duplicate)]
        receipt = executor(
            self._web_request(
                "tool.web_research",
                {"limit": 2, "query": "same", "sources": ["general"]},
            )
        )
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(len(receipt.observable_consequence.artifact_refs), 1)

    def test_human_library_read_continues_to_one_grounded_public_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            library_root = base / "library"
            (library_root / "catalog").mkdir(parents=True)
            works = {
                "works/first.txt": b"Alpha beta gamma.",
                "works/second.txt": b"Delta epsilon.",
            }
            items = []
            for index, (relative, body) in enumerate(works.items(), start=1):
                destination = library_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                items.append(
                    {
                        "path": relative,
                        "title": f"Synthetic Work {index}",
                        "author": "Synthetic Author",
                        "source": f"https://example.test/work-{index}",
                        "license_class": "test_only",
                        "default_reading_ingest": True,
                        "adapter_status": "test_not_for_adapter",
                    }
                )
            (library_root / "catalog/library.json").write_text(
                json.dumps(
                    {
                        "schema": "jenny.library.catalog.v1",
                        "acquired_at": "2026-09-03",
                        "jurisdiction_note": "Synthetic test sources only.",
                        "items": items,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (library_root / "catalog/SHA256SUMS").write_text(
                "\n".join(
                    f"{hashlib.sha256(body).hexdigest()}  {relative}"
                    for relative, body in sorted(works.items())
                )
                + "\n",
                encoding="utf-8",
            )
            executor = JennyLibraryExecutor(library_root)

            class CountingLibrary:
                def __init__(self, delegate):
                    self.delegate = delegate
                    self.calls = 0

                def __call__(self, request):
                    self.calls += 1
                    return self.delegate(request)

            counted = CountingLibrary(executor)
            controller = _HumanLibraryController()
            router_backend = _FullLibraryRouterBackend()
            cortex = _LibraryGroundedCortex()
            experience_model = _LibraryExperience()
            path = base / "human-library.sqlite3"
            binding = InternalAffordanceBinding(
                LIBRARY_AFFORDANCE,
                counted,
                executor.source_ref,
                LIBRARY_SOURCE_KIND,
            )
            runtime = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=experience_model,
                cortex=cortex,
                controller=controller,
                human_turn_router=AdaptiveHumanTurnRouter(router_backend),
                internal_affordance_bindings=(binding,),
            )
            self.assertIsNone(
                json.loads(runtime.supervisor.state_bytes())
                .get("library_reading_state", {})
                .get("catalog")
            )

            result = runtime.chat(
                "chat:library-visible",
                "Is your library empty? Check before answering.",
            )

            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(result.selected_affordance_id, "cortex.respond")
            self.assertEqual(
                runtime.turn_output(result.episode_ref),
                (
                    "I can see 2 eligible works in the verified catalog.",
                    "COMPLETED_UNEVALUATED",
                ),
            )
            episodes = runtime.supervisor.episode_items()
            self.assertEqual(len(episodes), 2)
            first, second = (json.loads(item.payload_json) for item in episodes)
            self.assertEqual(first["observation"]["source"], "HUMAN")
            self.assertEqual(
                first["choice"]["selected_affordance_id"],
                LIBRARY_AFFORDANCE_ID,
            )
            self.assertEqual(second["observation"]["source"], "CONTINUATION")
            self.assertEqual(
                second["choice"]["selected_affordance_id"], "cortex.respond"
            )
            self.assertEqual(second["previous_event_ref"], episodes[0].event_ref)
            self.assertEqual(counted.calls, 1)
            self.assertEqual(len(cortex.calls), 1)
            self.assertEqual(experience_model.outcome_assessment_calls, 0)
            self.assertEqual(
                [visible for _, visible in controller.calls],
                [],
            )

            request, _, _, _, cognitive_state = cortex.calls[0]
            self.assertEqual(
                request, "Is your library empty? Check before answering."
            )
            grounded = cognitive_state["library_turn_observation"]
            observable = grounded["observable_consequence"]
            catalog = observable["observation"]
            self.assertEqual(catalog["contract"], LIBRARY_CATALOG_CONTRACT)
            self.assertEqual(catalog["catalog_ref"], executor.catalog_ref)
            self.assertEqual(catalog["manifest_ref"], executor.manifest_ref)
            self.assertEqual(observable["source_ref"], executor.source_ref)
            self.assertEqual(
                grounded["library_observation_ref"], observable["observation_ref"]
            )
            self.assertEqual(grounded["library_episode_ref"], episodes[0].episode_ref)
            state_payload = json.loads(runtime.supervisor.state_bytes())
            last_human = state_payload["last_human_interaction"]
            self.assertEqual(
                last_human["human_observation"],
                "Is your library empty? Check before answering.",
            )
            self.assertEqual(
                last_human["model_output"],
                "I can see 2 eligible works in the verified catalog.",
            )
            self.assertEqual(
                last_human["observation_ref"],
                CycleObservation(**first["observation"]).observation_ref,
            )
            self.assertEqual(
                last_human["terminal_observation_ref"],
                CycleObservation(**second["observation"]).observation_ref,
            )

            replay = runtime.chat(
                "chat:library-visible",
                "Is your library empty? Check before answering.",
            )
            self.assertEqual(replay.episode_ref, result.episode_ref)
            self.assertEqual(len(runtime.supervisor.episode_items()), 2)
            self.assertEqual(counted.calls, 1)
            self.assertEqual(len(cortex.calls), 1)

            restarted_controller = _HumanLibraryController()
            restarted_router = _FullLibraryRouterBackend()
            restarted_cortex = _LibraryGroundedCortex()
            restarted = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_LibraryExperience(),
                cortex=restarted_cortex,
                controller=restarted_controller,
                human_turn_router=AdaptiveHumanTurnRouter(restarted_router),
                internal_affordance_bindings=(binding,),
            )
            restarted_result = restarted.chat(
                "chat:library-visible",
                "Is your library empty? Check before answering.",
            )
            self.assertEqual(restarted_result.episode_ref, result.episode_ref)
            self.assertEqual(restarted_controller.calls, [])
            self.assertEqual(restarted_router.calls, [])
            self.assertEqual(restarted_cortex.calls, [])
            self.assertEqual(counted.calls, 1)

            crash_path = base / "human-library-crash.sqlite3"
            crash_controller = _HumanLibraryController()
            crash_runtime = assemble_jenny2_runtime(
                crash_path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_LibraryExperience(),
                cortex=_LibraryGroundedCortex(),
                controller=crash_controller,
                human_turn_router=AdaptiveHumanTurnRouter(
                    _FullLibraryRouterBackend()
                ),
                internal_affordance_bindings=(binding,),
            )
            first_only = crash_runtime.supervisor.human_ingress(
                "chat:library-crash",
                "Is your library empty? Check before answering.",
            )
            self.assertEqual(
                first_only.selected_affordance_id, LIBRARY_AFFORDANCE_ID
            )
            self.assertEqual(len(crash_runtime.supervisor.episode_items()), 1)
            recovered_controller = _HumanLibraryController()
            recovered_cortex = _LibraryGroundedCortex()
            recovered = assemble_jenny2_runtime(
                crash_path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_LibraryExperience(),
                cortex=recovered_cortex,
                controller=recovered_controller,
                human_turn_router=AdaptiveHumanTurnRouter(
                    _FullLibraryRouterBackend()
                ),
                internal_affordance_bindings=(binding,),
            )
            recovered_result = recovered.chat(
                "chat:library-crash",
                "Is your library empty? Check before answering.",
            )
            self.assertEqual(
                recovered.turn_output(recovered_result.episode_ref)[0],
                "I can see 2 eligible works in the verified catalog.",
            )
            self.assertEqual(len(recovered.supervisor.episode_items()), 2)
            self.assertEqual(
                [visible for _, visible in recovered_controller.calls],
                [],
            )
            self.assertEqual(len(recovered_cortex.calls), 1)
            self.assertEqual(counted.calls, 2)

    def test_source_bound_internal_executor_commits_observation_once(self):
        source_ref = "sha256:" + "d" * 64

        class ObservedExecutor:
            def __init__(self):
                self.calls = 0
                self.last_request = None

            def __call__(self, request):
                self.calls += 1
                self.last_request = request
                observation = ObservableConsequence(
                    request_ref=request.idempotency_key,
                    source_kind="TEST",
                    source_ref=source_ref,
                    observation_json=(
                        '{"assertions_passed":2,"assertions_total":2,'
                        '"exit_code":0}'
                    ),
                    evidence_refs=("sha256:" + "e" * 64,),
                )
                return AffordanceReceipt(
                    "COMPLETED",
                    "Two of two source-bound assertions passed.",
                    (),
                    observation,
                )

        def map_observation(observation):
            measured = json.loads(observation.observation_json)
            passed = measured["assertions_passed"] == measured["assertions_total"]
            return ConsequenceVector(
                1.0 if passed else -1.0,
                1.0 if passed else -1.0,
                0.0 if passed else 1.0,
                0.5,
                1.0,
                0.5,
                0.0,
                1.0,
                0.0,
            )

        initial_state = json.dumps(
            {
                "affordance_signals": {},
                "affordance_utility": {"test.verify": 1.0},
                "memory_utility": {},
                "motivation_weights": {},
                "next_internal_request": "Run the bounded internal verification.",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        binding = lambda executor: InternalAffordanceBinding(
            Affordance(
                "test.verify",
                "ACT",
                "Run a bounded source-bound internal verification.",
                "internal.cognition",
            ),
            executor,
            source_ref,
            "TEST",
            map_observation,
            "sha256:" + "f" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed-internal.sqlite3"
            first_executor = ObservedExecutor()
            source = _Source()
            runtime = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=_Cortex(),
                selector_qualification_ref=QUALIFICATION,
                initial_state=initial_state,
                internal_affordance_bindings=(binding(first_executor),),
                clock=TrustedClock(
                    wall_clock=source.wall_now,
                    monotonic_clock=source.mono_now,
                ),
            )
            result = runtime.supervisor.scheduler_tick("test:observed-internal")
            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(first_executor.calls, 1)
            episode = json.loads(
                runtime.supervisor.episode_item(result.episode_ref).payload_json
            )
            observed = episode["receipt"]["observable_consequence"]
            self.assertEqual(observed["source_ref"], source_ref)
            self.assertEqual(
                observed["request_ref"],
                first_executor.last_request.idempotency_key,
            )
            self.assertEqual(episode["choice"]["selected_affordance_id"], "test.verify")
            self.assertTrue(runtime.supervisor.state_head().state_ref.startswith("sha256:"))

            second_executor = ObservedExecutor()
            restarted = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=_Cortex(),
                selector_qualification_ref=QUALIFICATION,
                initial_state=initial_state,
                internal_affordance_bindings=(binding(second_executor),),
            )
            replay = restarted.supervisor.human_ingress(
                "test:observed-internal", "Replay the already committed trigger."
            )
            self.assertEqual(replay.episode_ref, result.episode_ref)
            self.assertEqual(second_executor.calls, 0)

    def test_qwen38_library_root_assembles_one_exact_source_bound_affordance(self):
        source_ref = "sha256:" + "8" * 64
        library_executor = MagicMock()
        library_executor.source_ref = source_ref
        reference_backend = MagicMock()
        runtime = SimpleNamespace(supervisor=object(), _closers=())
        with tempfile.TemporaryDirectory() as directory, patch.object(
            jenny2_runtime_module,
            "JennyLibraryExecutor",
            return_value=library_executor,
        ) as executor_type, patch.object(
            jenny2_runtime_module,
            "ThreadedAsyncReferenceMemoryBackend",
            return_value=reference_backend,
        ), patch.object(
            jenny2_runtime_module,
            "assemble_jenny2_runtime",
            return_value=runtime,
        ) as assemble:
            library_root = Path(directory) / "synthetic-library"
            result = (
                jenny2_runtime_module.assemble_qwen38_autonomous_jenny2_with_cognee(
                    Path(directory) / "jenny.sqlite3",
                    genesis_created_at_utc="2026-09-03T00:00:00Z",
                    cognee_scope=object(),
                    runtime_ref="sha256:" + "7" * 64,
                    library_root=library_root,
                    enable_native_openclaw=False,
                )
            )

        self.assertIs(result, runtime)
        executor_type.assert_called_once_with(library_root)
        bindings = assemble.call_args.kwargs["internal_affordance_bindings"]
        matches = [
            binding
            for binding in bindings
            if binding.affordance.affordance_id
            == jenny2_runtime_module.LIBRARY_AFFORDANCE_ID
        ]
        self.assertEqual(len(matches), 1)
        binding = matches[0]
        self.assertIs(binding.affordance, jenny2_runtime_module.LIBRARY_AFFORDANCE)
        self.assertIs(binding.executor, library_executor)
        self.assertEqual(binding.observable_source_ref, source_ref)
        self.assertEqual(
            binding.observable_source_kind,
            jenny2_runtime_module.LIBRARY_SOURCE_KIND,
        )
        self.assertIsNone(binding.consequence_mapper)
        self.assertIsNone(binding.consequence_mapper_ref)

    def test_qwen38_runtime_owns_exactly_one_authored_artifact_affordance(self):
        authored_executor = MagicMock()
        reference_backend = MagicMock()
        runtime = SimpleNamespace(supervisor=object(), _closers=())
        with tempfile.TemporaryDirectory() as directory, patch.object(
            jenny2_runtime_module,
            "AuthoredArtifactExecutor",
            return_value=authored_executor,
        ) as executor_type, patch.object(
            jenny2_runtime_module,
            "QWEN38_NVFP4_MODEL_PATH",
            Path(directory),
        ), patch.object(
            jenny2_runtime_module,
            "ThreadedAsyncReferenceMemoryBackend",
            return_value=reference_backend,
        ), patch.object(
            jenny2_runtime_module,
            "assemble_jenny2_runtime",
            return_value=runtime,
        ) as assemble:
            result = (
                jenny2_runtime_module.assemble_qwen38_autonomous_jenny2_with_cognee(
                    Path(directory) / "jenny.sqlite3",
                    genesis_created_at_utc="2026-09-04T00:00:00Z",
                    cognee_scope=object(),
                    runtime_ref="sha256:" + "7" * 64,
                    enable_native_openclaw=False,
                )
            )

        self.assertIs(result, runtime)
        executor_type.assert_called_once_with()
        bindings = assemble.call_args.kwargs["internal_affordance_bindings"]
        matches = [
            binding
            for binding in bindings
            if binding.affordance.affordance_id
            == AUTHORED_ARTIFACT_AFFORDANCE_ID
        ]
        self.assertEqual(len(matches), 1)
        binding = matches[0]
        self.assertIs(binding.affordance, AUTHORED_ARTIFACT_AFFORDANCE)
        self.assertIs(binding.executor, authored_executor)
        self.assertEqual(binding.observable_source_ref, AUTHORED_ARTIFACT_SOURCE_REF)
        self.assertEqual(binding.observable_source_kind, AUTHORED_ARTIFACT_SOURCE_KIND)
        self.assertEqual(binding.affordance.permission_scope, "internal.cognition")
        self.assertFalse(binding.affordance.external_effect)
        self.assertIsNone(binding.consequence_mapper)
        self.assertIsNone(binding.consequence_mapper_ref)

        supplied_duplicate = InternalAffordanceBinding(
            affordance=AUTHORED_ARTIFACT_AFFORDANCE,
            executor=AuthoredArtifactExecutor(),
            observable_source_ref=AUTHORED_ARTIFACT_SOURCE_REF,
            observable_source_kind=AUTHORED_ARTIFACT_SOURCE_KIND,
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError,
            "authored-artifact affordance is runtime-owned",
        ):
            jenny2_runtime_module.assemble_qwen38_autonomous_jenny2_with_cognee(
                Path(directory) / "duplicate.sqlite3",
                genesis_created_at_utc="2026-09-04T00:00:00Z",
                cognee_scope=object(),
                runtime_ref="sha256:" + "6" * 64,
                internal_affordance_bindings=(supplied_duplicate,),
                enable_native_openclaw=False,
            )

    def test_qualitative_feedback_needs_no_scalar_and_changes_no_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            runtime = assemble_jenny2_runtime(
                Path(directory) / "qualitative-feedback.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=_Cortex(),
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            attempt = runtime.chat("chat:qualitative", "Try one contextual choice.")
            result = runtime.feedback(
                "feedback:qualitative",
                target_episode_ref=attempt.episode_ref,
                feedback_text="In this context, asking first would have exposed the missing preference.",
                feedback_source_ref="sha256:" + "9" * 64,
            )
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(runtime.supervisor.state_bytes())
            observation = learned["qualitative_feedback"][-1]
            self.assertEqual(
                observation["epistemic_status"],
                "HUMAN_QUALITATIVE_FEEDBACK_OBSERVATION",
            )
            self.assertIsNone(observation["consequence"])
            self.assertEqual(learned["affordance_utility"], {})
            self.assertNotIn("capability_evidence", learned)
            self.assertEqual(
                learned["compute_calibration"][-1]["epistemic_status"],
                "COMPUTE_ROUTE_WITH_QUALITATIVE_FEEDBACK",
            )

    def test_runtime_accepts_late_feedback_and_projects_the_learning_event(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            memory = _Memory()
            runtime = assemble_jenny2_runtime(
                Path(directory) / "feedback.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=memory,
                experience_model=_Experience(),
                cortex=_Cortex(),
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            attempt = runtime.chat("chat:unevaluated", "Try one bounded answer.")
            self.assertEqual(
                runtime.turn_output(attempt.episode_ref),
                ("Synthetic response.", "COMPLETED_UNEVALUATED"),
            )
            first_projection = memory.records[0][0]
            self.assertIn('"input_epistemic_status":"HUMAN_OBSERVATION"', first_projection)
            self.assertIn('"input":"Try one bounded answer."', first_projection)
            self.assertIn('"output":"Synthetic response."', first_projection)
            self.assertIn(
                '"output_epistemic_status":"MODEL_OUTPUT_UNEVALUATED"',
                first_projection,
            )
            initial_state = runtime.status().state_ref
            feedback = runtime.feedback(
                "feedback:one",
                target_episode_ref=attempt.episode_ref,
                feedback_text="The answer was useful but could be more specific.",
                feedback_source_ref="sha256:" + "5" * 64,
                consequence=ConsequenceVector(
                    0.6, 0.8, 0.3, 0.4, 0.7, 0.5, 0.1, 1.0, 0.7
                ),
            )
            self.assertEqual(feedback.status, "COMMITTED")
            self.assertEqual(runtime.status().moving_origin_ordinal, 1)
            self.assertTrue(runtime.status().moving_origin_time_utc.endswith("Z"))
            self.assertIn("-04:00", runtime.status().moving_origin_local_time)
            self.assertEqual(
                runtime.status().moving_origin_timezone, "America/New_York"
            )
            self.assertNotEqual(runtime.status().state_ref, initial_state)
            learned = json.loads(runtime.supervisor.state_bytes())
            feedback_observation = learned["qualitative_feedback"][-1]
            self.assertEqual(
                feedback_observation["epistemic_status"],
                "HUMAN_FEEDBACK_OBSERVATION",
            )
            self.assertEqual(
                feedback_observation["feedback_text"],
                "The answer was useful but could be more specific.",
            )
            self.assertEqual(
                feedback_observation["target_episode_ref"], attempt.episode_ref
            )
            self.assertEqual(
                learned["compute_calibration"][-1]["epistemic_status"],
                "COMPUTE_ROUTE_WITH_OBSERVED_CONSEQUENCE",
            )
            self.assertEqual(feedback_observation["consequence"]["human_feedback"], 0.7)
            self.assertEqual(len(memory.records), 2)
            self.assertEqual(runtime.status().pending_projections, 0)

    def test_assembled_runtime_commits_projects_restarts_and_runs_life(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            memory, cortex = _Memory(), _Cortex()
            runtime = assemble_jenny2_runtime(
                Path(directory) / "jenny2.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=memory,
                experience_model=_Experience(),
                cortex=cortex,
                consequence_evaluator=_evaluate,
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            chat = runtime.chat("chat:one", "A bounded synthetic request.")
            self.assertEqual(chat.status, "COMMITTED")
            self.assertEqual(
                [item.affordance_id for item in runtime.supervisor.affordances.definitions()],
                ["cortex.respond"],
            )
            self.assertEqual(len(memory.records), 1)
            self.assertEqual(runtime.status().pending_projections, 0)
            runtime.start_life(interval_seconds=0.001, max_steps_per_session=2)
            life = runtime.wait_life(timeout=2.0)
            self.assertEqual([item.status for item in life], ["QUIESCENT", "QUIESCENT"])
            self.assertTrue(
                all(item.moving_origin_ordinal == chat.moving_origin_ordinal for item in life)
            )
            self.assertEqual(runtime.status().moving_origin_ordinal, 0)
            self.assertEqual(cortex.calls, 1)
            self.assertEqual(len(memory.records), 1)
            activity = runtime.activity()
            self.assertEqual(activity.phase, "IDLE")
            self.assertEqual(activity.current_status, "QUIESCENT")
            self.assertEqual(activity.last_completed["episode_ref"], chat.episode_ref)
            runtime.supervisor.set_scheduler_enabled(False)
            self.assertFalse(runtime.status().scheduler_enabled)
            still_chat = runtime.chat("chat:two", "Chat remains available.")
            self.assertEqual(still_chat.status, "COMMITTED")
            self.assertEqual(cortex.calls, 2)

    def test_activity_exposes_inference_phase_and_restores_last_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activity.sqlite3"
            source = _Source()
            cortex = _BlockingCortex()
            runtime = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=cortex,
                consequence_evaluator=_evaluate,
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            outcomes = []
            worker = threading.Thread(
                target=lambda: outcomes.append(
                    runtime.chat("activity:one", "A bounded synthetic request.")
                )
            )
            worker.start()
            self.assertTrue(cortex.entered.wait(timeout=2))
            during = runtime.activity()
            self.assertEqual(during.phase, "EXECUTE")
            self.assertEqual(during.operation_source, "HUMAN")
            self.assertEqual(during.current_status, "RUNNING")
            self.assertIsNone(during.last_completed)
            cortex.release.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(outcomes), 1)
            committed = runtime.activity()
            self.assertEqual(committed.phase, "IDLE")
            self.assertEqual(committed.current_status, "COMMITTED")
            self.assertEqual(
                committed.last_completed["episode_ref"], outcomes[0].episode_ref
            )

            restarted = assemble_jenny2_runtime(
                path,
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=_Cortex(),
                consequence_evaluator=_evaluate,
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            restored = restarted.activity()
            self.assertEqual(restored.phase, "IDLE")
            self.assertEqual(restored.current_status, "COMMITTED")
            self.assertEqual(restored.last_completed, committed.last_completed)

    def test_model_authored_pending_operation_drives_bounded_internal_life(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            cortex = _Cortex()
            initial_state = json.dumps(
                {
                    "affordance_signals": {},
                    "affordance_utility": {},
                    "memory_utility": {},
                    "motivation_weights": {},
                    "next_internal_request": "Examine one bounded unresolved pattern.",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            runtime = assemble_jenny2_runtime(
                Path(directory) / "pending-operation.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=cortex,
                consequence_evaluator=_evaluate,
                selector_qualification_ref=QUALIFICATION,
                initial_state=initial_state,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            runtime.start_life(interval_seconds=0.001, max_steps_per_session=2)
            life = runtime.wait_life(timeout=2.0)
            self.assertEqual([item.status for item in life], ["COMMITTED", "COMMITTED"])
            self.assertEqual(cortex.calls, 2)
            self.assertEqual(runtime.status().moving_origin_ordinal, 1)

    def test_unqualified_assembled_runtime_is_truthfully_shadow_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            memory, cortex = _Memory(), _Cortex()
            runtime = assemble_jenny2_runtime(
                Path(directory) / "shadow.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=memory,
                experience_model=_Experience(),
                cortex=cortex,
                consequence_evaluator=_evaluate,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            result = runtime.chat("chat:shadow", "Do not overclaim readiness.")
            self.assertEqual(result.status, "SHADOW")
            self.assertEqual(cortex.calls, 0)
            self.assertEqual(runtime.status().moving_origin_ordinal, -1)

    def test_canonical_memory_fallback_rejoins_the_one_episode_store(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _Source()
            runtime = assemble_jenny2_runtime(
                Path(directory) / "canonical-memory.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=None,
                experience_model=_Experience(),
                cortex=_Cortex(),
                consequence_evaluator=_evaluate,
                selector_qualification_ref=QUALIFICATION,
                clock=TrustedClock(
                    wall_clock=source.wall_now, monotonic_clock=source.mono_now
                ),
            )
            runtime.chat("memory:first", "Verify one bounded consequence.")
            recalled = runtime.supervisor.cycle.memory.recall(
                "verified consequence", limit=4
            )
            self.assertEqual(len(recalled), 1)
            self.assertEqual(recalled[0].acquired_ordinal, 0)
            self.assertEqual(runtime.status().pending_projections, 0)

    def test_native_tool_turn_commits_once_only_after_the_final_model_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _Source()
            memory = _Memory()
            cortex = _Cortex()
            controller = _NativeController()
            model = _NativeModel()
            tools = _native_tools()
            catalog = NativeOpenClawCatalog(tools)
            runtime = assemble_jenny2_runtime(
                root / "native-runtime.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=memory,
                experience_model=_Experience(),
                cortex=cortex,
                controller=controller,
                native_tool_model=model,
                native_turn_path=root / "native-turn.sqlite3",
                clock=TrustedClock(
                    wall_clock=source.wall_now,
                    monotonic_clock=source.mono_now,
                ),
            )
            initial_head = runtime.supervisor.state_head()

            requested = runtime.native_chat(
                "owner:native-one",
                "Find the source-bound result, then answer me.",
                tools=tools,
                catalog_hash=catalog.catalog_hash,
            )

            self.assertEqual(requested.status, "TOOL_REQUESTED")
            self.assertIsNotNone(requested.turn_id)
            self.assertIsNotNone(requested.turn_revision)
            self.assertEqual(requested.moving_origin_ordinal, -1)
            self.assertEqual(runtime.supervisor.state_head().state_ref, initial_head.state_ref)
            self.assertEqual(runtime.supervisor.state_head().moving_origin_ordinal, -1)
            self.assertEqual(runtime.supervisor.episode_items(), ())
            self.assertEqual(runtime.supervisor.pending_projections(limit=256), ())
            self.assertEqual(memory.records, [])
            self.assertEqual(cortex.calls, 0)
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(model.calls[0]["prior_trace_json"], "[]")
            self.assertEqual(model.calls[0]["catalog_hash"], catalog.catalog_hash)
            self.assertEqual(
                tuple(model.calls[0]["tools"]),
                tuple(card.provider_payload() for card in catalog.cards),
            )
            self.assertFalse(runtime.status().external_effects_enabled)

            pending = runtime.supervisor.pending_deferred_effect()
            self.assertIsNotNone(pending)
            pending_affordance, pending_request = pending
            self.assertEqual(
                pending_affordance.affordance_id, "openclaw.supervised.turn"
            )
            self.assertEqual(pending_affordance.authorization_mode, "HOST_GUARDED")
            call = requested.tool_calls[0]
            self.assertEqual(
                requested.permission_reservation_refs,
                (call["permission_reservation_ref"],),
            )

            committed = runtime.native_continue(
                turn_id=requested.turn_id,
                turn_revision=requested.turn_revision,
                tool_results=(
                    NativeHostToolResult(
                        call_id=call["id"],
                        tool_name=call["function"]["name"],
                        operation_ref=call["operation_ref"],
                        permission_reservation_ref=call[
                            "permission_reservation_ref"
                        ],
                        status="COMPLETED",
                        result={"answer": 42, "source": "guarded-host"},
                    ),
                ),
            )

            self.assertEqual(committed.status, "COMMITTED")
            self.assertEqual(committed.message, "The host-observed result is forty-two.")
            self.assertEqual(committed.moving_origin_ordinal, 0)
            self.assertIsNotNone(committed.episode_ref)
            self.assertEqual(len(model.calls), 2)
            self.assertIn('"kind":"TOOL_RESULT"', model.calls[1]["prior_trace_json"])
            self.assertEqual(runtime.supervisor.state_head().moving_origin_ordinal, 0)
            self.assertNotEqual(runtime.supervisor.state_head().state_ref, initial_head.state_ref)
            self.assertEqual(len(runtime.supervisor.episode_items()), 1)
            self.assertEqual(len(memory.records), 1)
            self.assertEqual(runtime.status().pending_projections, 0)
            self.assertFalse(runtime.status().pending_operation)
            self.assertFalse(runtime.status().external_effects_enabled)
            self.assertEqual(controller.update_calls, 0)

            turn = runtime._native_turn_store.get(requested.turn_id)
            self.assertEqual(turn.status, "COMMITTED")
            self.assertEqual(turn.committed_episode_ref, committed.episode_ref)
            self.assertEqual(
                turn.permission_reservation_refs,
                requested.permission_reservation_refs,
            )
            result_trace = tuple(
                item for item in turn.trace if item.get("kind") == "TOOL_RESULT"
            )
            self.assertEqual(len(result_trace), 1)
            episode = json.loads(
                runtime.supervisor.episode_item(committed.episode_ref).payload_json
            )
            self.assertEqual(episode["observation"]["source"], "AGENT")
            self.assertEqual(
                episode["choice"]["selected_affordance_id"],
                "openclaw.supervised.turn",
            )
            observed = episode["receipt"]["observable_consequence"]
            self.assertEqual(observed["request_ref"], pending_request.idempotency_key)
            self.assertEqual(observed["source_ref"], catalog.catalog_hash)
            self.assertEqual(observed["artifact_refs"], [call["operation_ref"]])
            self.assertEqual(
                observed["evidence_refs"], [result_trace[0]["event_ref"]]
            )
            aggregate = json.loads(observed["observation_json"])
            self.assertEqual(aggregate["catalog_hash"], catalog.catalog_hash)
            self.assertEqual(aggregate["turn_id"], requested.turn_id)
            self.assertEqual(
                aggregate["tool_results"][0]["operation_ref"],
                call["operation_ref"],
            )
            self.assertEqual(
                aggregate["tool_results"][0]["event_ref"],
                result_trace[0]["event_ref"],
            )

            replay = runtime.native_chat(
                "owner:native-one",
                "Find the source-bound result, then answer me.",
                tools=tools,
                catalog_hash=catalog.catalog_hash,
            )
            self.assertEqual(replay.status, "COMMITTED")
            self.assertEqual(replay.episode_ref, committed.episode_ref)
            self.assertEqual(replay.message, committed.message)
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(len(runtime.supervisor.episode_items()), 1)
            self.assertEqual(len(memory.records), 1)

    def test_native_multi_tool_batch_is_atomic_and_old_revision_replay_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _Source()
            memory = _Memory()
            cortex = _Cortex()
            model = _TwoToolNativeModel()
            tools = _native_two_tools()
            catalog = NativeOpenClawCatalog(tools)
            runtime = assemble_jenny2_runtime(
                root / "native-batch-runtime.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=memory,
                experience_model=_Experience(),
                cortex=cortex,
                controller=_NativeController(),
                native_tool_model=model,
                native_turn_path=root / "native-batch-turn.sqlite3",
                clock=TrustedClock(
                    wall_clock=source.wall_now,
                    monotonic_clock=source.mono_now,
                ),
            )
            requested = runtime.native_chat(
                "owner:native-batch",
                "Use both guarded observations before answering.",
                tools=tools,
                catalog_hash=catalog.catalog_hash,
            )
            self.assertEqual(requested.status, "TOOL_REQUESTED")
            self.assertEqual(len(requested.tool_calls), 2)
            first_call, second_call = requested.tool_calls

            def host_result(call, result):
                return NativeHostToolResult(
                    call_id=call["id"],
                    tool_name=call["function"]["name"],
                    operation_ref=call["operation_ref"],
                    permission_reservation_ref=call[
                        "permission_reservation_ref"
                    ],
                    status="COMPLETED",
                    result=result,
                )

            first_result = host_result(
                first_call, {"answer": "ordinal time is event-relative"}
            )
            second_result = host_result(
                second_call, {"content": "the resource confirms the observation"}
            )
            malformed_second = NativeHostToolResult(
                call_id=second_result.call_id,
                tool_name=second_result.tool_name,
                operation_ref=first_result.operation_ref,
                permission_reservation_ref=second_result.permission_reservation_ref,
                status=second_result.status,
                result=dict(second_result.result),
            )

            with self.assertRaisesRegex(ValueError, "provenance differs"):
                runtime.native_continue(
                    turn_id=requested.turn_id,
                    turn_revision=requested.turn_revision,
                    tool_results=(first_result, malformed_second),
                )
            untouched = runtime._native_turn_store.get(requested.turn_id)
            self.assertEqual(untouched.revision, requested.turn_revision)
            self.assertEqual(len(untouched.outstanding_calls), 2)
            self.assertFalse(
                any(item.get("kind") == "TOOL_RESULT" for item in untouched.trace)
            )
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(runtime.supervisor.episode_items(), ())
            self.assertEqual(runtime.supervisor.state_head().moving_origin_ordinal, -1)

            committed = runtime.native_continue(
                turn_id=requested.turn_id,
                turn_revision=requested.turn_revision,
                tool_results=(first_result, second_result),
            )
            self.assertEqual(committed.status, "COMMITTED")
            self.assertEqual(committed.moving_origin_ordinal, 0)
            self.assertEqual(committed.turn_revision, requested.turn_revision + 3)
            self.assertEqual(len(model.calls), 2)
            stored = runtime._native_turn_store.get(requested.turn_id)
            recorded_results = tuple(
                item for item in stored.trace if item.get("kind") == "TOOL_RESULT"
            )
            self.assertEqual(len(recorded_results), 2)
            self.assertEqual(
                {item["call_id"] for item in recorded_results},
                {first_result.call_id, second_result.call_id},
            )
            self.assertEqual(len(runtime.supervisor.episode_items()), 1)
            self.assertEqual(len(memory.records), 1)

            immediate_replay = runtime.native_continue(
                turn_id=requested.turn_id,
                turn_revision=requested.turn_revision,
                tool_results=(first_result, second_result),
            )
            self.assertEqual(immediate_replay, committed)
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(len(runtime.supervisor.episode_items()), 1)
            self.assertEqual(len(memory.records), 1)

            later = runtime.chat(
                "owner:ordinary-after-native",
                "Record one later ordinary episode.",
            )
            self.assertEqual(later.status, "COMMITTED")
            self.assertEqual(later.moving_origin_ordinal, 1)
            self.assertEqual(cortex.calls, 1)
            self.assertEqual(len(runtime.supervisor.episode_items()), 2)
            self.assertEqual(len(memory.records), 2)

            later_replay = runtime.native_continue(
                turn_id=requested.turn_id,
                turn_revision=requested.turn_revision,
                tool_results=(first_result, second_result),
            )
            self.assertEqual(later_replay.episode_ref, committed.episode_ref)
            self.assertEqual(later_replay.message, committed.message)
            self.assertEqual(later_replay.turn_id, committed.turn_id)
            self.assertEqual(later_replay.turn_revision, committed.turn_revision)
            self.assertEqual(later_replay.moving_origin_ordinal, 0)
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(len(runtime.supervisor.episode_items()), 2)
            self.assertEqual(len(memory.records), 2)

            divergent_second = host_result(
                second_call, {"content": "a different retry result"}
            )
            with self.assertRaisesRegex(RuntimeError, "revision conflict"):
                runtime.native_continue(
                    turn_id=requested.turn_id,
                    turn_revision=requested.turn_revision,
                    tool_results=(first_result, divergent_second),
                )
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(len(runtime.supervisor.episode_items()), 2)
            self.assertEqual(len(memory.records), 2)

    def test_native_host_guarded_affordance_is_visible_only_to_agent_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _Source()
            controller = _NativeController()
            tools = _native_tools()
            catalog = NativeOpenClawCatalog(tools)
            initial_state = json.dumps(
                {
                    "affordance_signals": {},
                    "affordance_utility": {},
                    "memory_utility": {},
                    "motivation_weights": {},
                    "next_internal_request": "Inspect one learned pending concern.",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            runtime = assemble_jenny2_runtime(
                root / "native-visibility.sqlite3",
                genesis_created_at_utc="2026-09-02T12:00:00Z",
                memory=_Memory(),
                experience_model=_Experience(),
                cortex=_Cortex(),
                controller=controller,
                initial_state=initial_state,
                native_tool_model=_NativeModel(),
                native_turn_path=root / "native-visibility-turn.sqlite3",
                clock=TrustedClock(
                    wall_clock=source.wall_now,
                    monotonic_clock=source.mono_now,
                ),
            )
            runtime.configure_native_openclaw(
                tools, catalog_hash=catalog.catalog_hash
            )
            definitions = runtime.supervisor.affordances.definitions()
            guarded = runtime.supervisor.affordances.definition(
                "openclaw.supervised.turn"
            )
            self.assertEqual(guarded.authorization_mode, "HOST_GUARDED")

            runtime.supervisor.cycle.choose(
                observation=CycleObservation(
                    "visibility:scheduler", "SCHEDULER", ""
                ),
                temporal=runtime.supervisor.clock.sample(-1),
                affordances=definitions,
                state=runtime.supervisor.state_bytes(),
            )
            agent_choice = runtime.supervisor.cycle.choose(
                observation=CycleObservation(
                    "visibility:agent", "AGENT", "Consider a guarded search."
                ),
                temporal=runtime.supervisor.clock.sample(-1),
                affordances=definitions,
                state=runtime.supervisor.state_bytes(),
            )

            scheduler_visible = controller.visible_by_source["SCHEDULER"][0]
            agent_visible = controller.visible_by_source["AGENT"][0]
            self.assertNotIn("openclaw.supervised.turn", scheduler_visible)
            self.assertIn("openclaw.supervised.turn", agent_visible)
            self.assertEqual(
                agent_choice.selected_affordance_id, "openclaw.supervised.turn"
            )
            self.assertEqual(runtime.supervisor.episode_items(), ())
            self.assertFalse(runtime.status().external_effects_enabled)


if __name__ == "__main__":
    unittest.main()
