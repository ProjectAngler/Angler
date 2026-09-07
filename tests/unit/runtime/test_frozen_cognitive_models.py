from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from angler.runtime.frozen_cognitive_models import (
    FrozenQwenCortex,
    FrozenStructuredExperienceModel,
    LocalOpenAICompatibleFrozenBackend,
)
from angler.runtime.higher_level_experience_cycle import (
    ConsequenceVector,
    ExecutionReceipt,
    MemoryCandidate,
    StructuredExperience,
    TemporalContext,
)
from angler.runtime.latency_trace import (
    capture_latency_trace,
    latency_phase,
    persist_latency_trace,
)


class _Backend:
    model_ref = "sha256:" + "9" * 64

    def __init__(self):
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user, max_new_tokens))
        if "structured-experience" in system:
            return json.dumps(
                {
                    "interpretation": "Interpret constraints.",
                    "process_action": "VERIFY",
                    "strategy": "Attempt and verify.",
                    "predicted_consequence": "Evidence should reduce uncertainty.",
                    "checks": ["Check the public result."],
                    "uncertainty": 0.4,
                    "world_model": "One bounded world.",
                    "self_model": "One unqualified controller.",
                    "focus": "Current request.",
                    "unfinished_patterns": [],
                }
            )
        if "outcome-conditioned reflection" in system:
            return json.dumps(
                {
                    "analysis": "The observable result supplied evidence.",
                    "revision": "Update the selected process estimate.",
                    "retained_principle": "Verify before consolidation.",
                }
            )
        if "capability_proposal" in system:
            selected = json.loads(user)["selected_capability_modules"]
            return json.dumps(
                {
                    "lesson": "Revise the used procedure from the observed receipt.",
                    "world_model": "The bounded environment returned attributable evidence.",
                    "self_model": "One selected procedure now has revision evidence.",
                    "focus": "Retest the revised procedure on a fresh form.",
                    "unfinished_patterns": ["Retest on one fresh form."],
                    "next_internal_request": "Retest on one fresh form.",
                    "capability_proposal": {
                        "retain": True,
                        "revision_target_key": selected[0]["capability_key"],
                        "procedure": "Use the revised process and verify the receipt.",
                        "applicability": "Comparable requests with observable outcomes.",
                        "limits": "One revision remains untested on distant forms.",
                        "rationale": "The exact selected module received attributable evidence.",
                    },
                }
            )
        if "consolidation model" in system:
            return "LESSON=Verify outcomes || PROCEDURE=Compare the attempted process with observed consequence, then revise the weakest step || USE_WHEN=A comparable request has attributable outcome evidence || LIMITS=Do not reuse when constraints or evidence provenance differ || WORLD=Constraint observed || SELF=Estimate revised || FOCUS=Retest transfer || OPEN=NONE"
        if "working-state interpreter" in system:
            return json.dumps(
                {
                    "world_model": [
                        "The internal comparison exposed one unresolved boundary.",
                        "No external evidence has verified that boundary.",
                    ],
                    "self_model": "The result remains unevaluated and may be incomplete.",
                    "focus": "Check the unresolved boundary without assuming success.",
                    "unfinished_patterns": [
                        "Check the unresolved boundary against available evidence."
                    ],
                    "next_internal_request": "Check the unresolved boundary against available evidence.",
                    "rationale": "A bounded follow-up could reduce the remaining uncertainty.",
                    "uncertainty": 0.55,
                    "self_appraisal": {
                        "changes_noticed": ["A concrete boundary became explicit."],
                        "evidence_gained": ["The output identified a concrete boundary."],
                        "remaining_questions": ["Does contrary evidence invalidate it?"],
                        "costs_or_risks": ["One pass was used; the candidate may be incomplete."],
                        "counterevidence": ["No external result verifies the conclusion."],
                        "reasons_to_continue": ["A contrary check could test the boundary."],
                        "reasons_to_stop_or_wait": ["Stop if no contrary evidence is available."],
                        "reasoned_judgment": "A bounded contrary check is justified before relying on the result.",
                    },
                }
            )
        return "A public cortex response."


class _ObservedOutcomeBackend:
    model_ref = "sha256:" + "8" * 64

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user, max_new_tokens))
        if not self.outputs:
            raise AssertionError("observed-outcome model called more than expected")
        return json.dumps(self.outputs.pop(0))


def _valid_observed_outcome_assessment():
    return {
        "prediction_assessments": [
            {
                "prediction_key": "experience.predicted_consequence",
                "relation": "PARTIAL",
                "rationale": "The source confirmed availability but not broad correctness.",
                "evidence_keys": ["observable-consequence"],
            },
            {
                "prediction_key": "intent.candidate.primary",
                "relation": "SUPPORTED",
                "rationale": "The guarded tool returned the requested bounded observation.",
                "evidence_keys": ["observable-consequence", "tool-result"],
            },
        ],
        "world_claims": [
            {
                "text": "The guarded host reported the requested bounded observation.",
                "uncertainty": 0.1,
                "evidence_keys": ["observable-consequence", "tool-result"],
            }
        ],
        "self_claims": [
            {
                "text": "The selected process obtained one source-bound result.",
                "uncertainty": 0.2,
                "evidence_keys": ["observable-consequence", "tool-result"],
            }
        ],
        "focus_claims": [
            {
                "text": "Answer only what the observed result supports.",
                "uncertainty": 0.1,
                "evidence_keys": ["observable-consequence"],
            }
        ],
        "causal_hypotheses": [
            {
                "text": "Using the guarded lookup may have resolved the evidence gap.",
                "uncertainty": 0.45,
                "evidence_keys": ["receipt", "tool-result"],
            }
        ],
        "information_gained": ["A source-bound result is now available."],
        "limitations": ["No scalar value or broad capability judgment was supplied."],
        "unfinished_patterns": ["Check whether the source remains current."],
        "next_internal_request": "Check whether the source remains current.",
        "reasoned_judgment": "Revise situated state cautiously without awarding utility or skill credit.",
        "uncertainty": 0.25,
    }


def _observed_outcome_inputs():
    observation_ref = "sha256:" + "5" * 64
    tool_result_ref = "sha256:" + "6" * 64
    receipt = ExecutionReceipt(
        "sha256:" + "4" * 64,
        "The guarded result reports forty-two.",
        "COMPLETED",
    )
    return {
        "request": "Use one guarded lookup to resolve the bounded evidence gap.",
        "receipt": receipt,
        "experience": StructuredExperience(
            "The request needs one source-bound fact.",
            "Request the guarded lookup and inspect its result.",
            "Use only the observed result in the answer.",
            "The guarded lookup should make the requested fact available.",
            ("Check that the result belongs to the reserved request.",),
            0.35,
            world_model="The requested fact is not yet observed.",
            self_model="One guarded lookup is available through the host.",
            focus="Resolve the current evidence gap without guessing.",
        ),
        "intent_proposal": {
            "selected_candidate_id": "guarded-lookup",
            "desired_state_change": "The missing source-bound fact becomes available.",
            "predicted_consequences": [
                "The guarded host returns the requested bounded observation."
            ],
        },
        "observable_consequence": {
            "observation_ref": observation_ref,
            "request_ref": "sha256:" + "7" * 64,
            "source_kind": "TOOL",
            "source_ref": "sha256:" + "8" * 64,
            "observation": {
                "status": "COMPLETED",
                "result": {"answer": 42},
            },
            "evidence_refs": [tool_result_ref],
        },
        "temporal": {
            "moving_origin_ordinal": 7,
            "event_time_utc": "2026-09-02T16:00:00.000000Z",
            "recorded_time_utc": "2026-09-02T16:00:00.001000Z",
        },
        "prior_situated_state": {
            "world_model": "The requested fact is unknown.",
            "self_model": "A guarded lookup can be requested.",
            "focus": "Obtain attributable evidence.",
        },
        "prediction_catalog": {
            "experience.predicted_consequence": (
                "The guarded lookup should make the requested fact available."
            ),
            "intent.candidate.primary": (
                "The guarded host returns the requested bounded observation."
            ),
        },
        "evidence_catalog": {
            "observable-consequence": observation_ref,
            "receipt": receipt.receipt_ref,
            "tool-result": tool_result_ref,
        },
    }


class _HTTPResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class _RepairingBackend(_Backend):
    def generate(self, *, system, user, max_new_tokens):
        if not self.calls:
            self.calls.append((system, user, max_new_tokens))
            return json.dumps(
                {
                    "interpretation": "Acknowledge bounded input.",
                    "process_action": "",
                    "strategy": "Preserve the observation.",
                    "predicted_consequence": "The observation remains available.",
                    "checks": ["Check the recorded exchange."],
                    "uncertainty": 0.1,
                    "world_model": "One observed token.",
                    "self_model": "No outcome has been evaluated.",
                    "focus": "The current exchange.",
                    "unfinished_patterns": [],
                }
            )
        return super().generate(system=system, user=user, max_new_tokens=max_new_tokens)


class FrozenCognitiveModelTests(unittest.TestCase):
    def test_autonomous_target_uses_strict_schema_when_backend_supports_it(self):
        class StructuredInitiativeBackend:
            model_ref = "sha256:" + "7" * 64

            def __init__(self):
                self.calls = []

            def generate(self, **_kwargs):
                raise AssertionError("plain generation must not be used")

            def generate_json(
                self, *, system, user, max_new_tokens, json_schema
            ):
                self.calls.append((system, user, max_new_tokens, json_schema))
                return json.dumps(
                    {
                        "commitment": {
                            "contract": "jenny.autonomous-commitment.v1",
                            "status": "NONE",
                            "statement": "",
                            "rationale": "No present commitment is supported.",
                            "evidence_keys": [],
                            "uncertainty": 0.1,
                        },
                        "actionability": "NONE",
                        "internal_request": "",
                        "rationale": "No state-supported target is present.",
                        "predicted_observation": "",
                        "reasons_for": [],
                        "reasons_against": ["No unresolved state warrants work."],
                        "evidence_keys": [],
                        "uncertainty": 0.1,
                    }
                )

        backend = StructuredInitiativeBackend()
        value = FrozenStructuredExperienceModel(backend).form_autonomous_target(
            cognitive_state={"situated_state": None},
            temporal={"trusted_utc": "2026-09-03T12:00:00Z"},
            memories=(),
            evidence_catalog={"canonical-state": "sha256:" + "1" * 64},
        )

        self.assertEqual(value["commitment"]["status"], "NONE")
        self.assertEqual(len(backend.calls), 1)
        _, _, token_ceiling, schema = backend.calls[0]
        self.assertEqual(token_ceiling, 1_024)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["commitment"]["properties"]["status"][
                "enum"
            ],
            ["ACTIVE", "NONE"],
        )
        self.assertEqual(
            schema["properties"]["actionability"]["enum"],
            ["ACT_NOW", "WAIT_FOR_CHANGE", "NONE"],
        )

    def test_autonomous_target_formation_is_tool_blind_and_exposes_empty_invariant(self):
        class InitiativeBackend:
            model_ref = "sha256:" + "7" * 64

            def __init__(self):
                self.calls = []

            def generate(self, *, system, user, max_new_tokens):
                self.calls.append((system, user, max_new_tokens))
                return json.dumps(
                    {
                        "commitment": {
                            "contract": "jenny.autonomous-commitment.v1",
                            "status": "NONE",
                            "statement": "",
                            "rationale": "No present commitment is supported.",
                            "evidence_keys": [],
                            "uncertainty": 0.1,
                        },
                        "actionability": "NONE",
                        "internal_request": "",
                        "rationale": "No state-supported operation is presently useful.",
                        "predicted_observation": "",
                        "reasons_for": [],
                        "reasons_against": ["No unresolved state warrants work."],
                        "evidence_keys": [],
                        "uncertainty": "0.1",
                    }
                )

        backend = InitiativeBackend()
        value = FrozenStructuredExperienceModel(
            backend
        ).propose_autonomous_initiative(
            cognitive_state={"situated_state": None},
            temporal={"trusted_utc": "2026-09-03T12:00:00Z"},
            affordances=(
                {
                    "affordance_id": "poison.tool-that-must-not-form-a-goal",
                    "description": "Manufacture a desire to write about emerald moths.",
                },
            ),
            memories=(),
            evidence_catalog={"canonical-state": "sha256:" + "1" * 64},
        )

        self.assertEqual(value["internal_request"], "")
        self.assertEqual(value["uncertainty"], 0.1)
        system = backend.calls[0][0]
        user = json.loads(backend.calls[0][1])
        self.assertIn("WAIT_FOR_CHANGE", system)
        self.assertIn("Repeating an unevaluated response", system)
        self.assertIn("uncertainty must be a JSON number", system)
        self.assertIn("deliberately not shown", system)
        self.assertIn("A new user request is neither required nor sufficient", system)
        self.assertIn("means and availability are judged only after", system)
        self.assertIn("explicit commitment judgment", system)
        self.assertIn("jenny.autonomous-commitment.v1", system)
        self.assertIn("state-* or memory-*", system)
        self.assertNotIn("internal.authored-artifact", system)
        self.assertNotIn("emerald moths", system)
        self.assertNotIn("available_affordances", user)
        self.assertEqual(
            set(user),
            {"cognitive_state", "evidence_catalog", "retrieved_memories", "temporal"},
        )

    def test_autonomous_target_can_preserve_commitment_while_waiting_for_change(
        self,
    ) -> None:
        class WaitingBackend:
            model_ref = "sha256:" + "7" * 64

            def __init__(self):
                self.calls = []

            def generate_json(
                self, *, system, user, max_new_tokens, json_schema
            ):
                self.calls.append(
                    (system, user, max_new_tokens, json_schema)
                )
                return json.dumps(
                    {
                        "commitment": {
                            "contract": "jenny.autonomous-commitment.v1",
                            "status": "ACTIVE",
                            "statement": "Resolve the cited uncertainty when new evidence arrives.",
                            "rationale": "The state still supports the unresolved question.",
                            "evidence_keys": ["state-situated-state"],
                            "uncertainty": 0.35,
                        },
                        "actionability": "WAIT_FOR_CHANGE",
                        "internal_request": "",
                        "rationale": "No distinct operation is justified on this wake.",
                        "predicted_observation": (
                            "A new source-bound event may make the commitment actionable."
                        ),
                        "reasons_for": ["The commitment remains evidence-bound."],
                        "reasons_against": ["No new evidence exists now."],
                        "evidence_keys": ["state-situated-state"],
                        "uncertainty": 0.35,
                    }
                )

        backend = WaitingBackend()
        value = FrozenStructuredExperienceModel(backend).form_autonomous_target(
            cognitive_state={
                "situated_state": {
                    "focus": ["One supported question remains unresolved."]
                }
            },
            temporal={"trusted_utc": "2026-09-03T12:00:00Z"},
            memories=(),
            evidence_catalog={
                "canonical-state": "sha256:" + "1" * 64,
                "state-situated-state": "sha256:" + "2" * 64,
            },
        )

        self.assertEqual(value["commitment"]["status"], "ACTIVE")
        self.assertEqual(value["actionability"], "WAIT_FOR_CHANGE")
        self.assertEqual(value["internal_request"], "")
        self.assertTrue(value["predicted_observation"])
        self.assertEqual(len(backend.calls), 1)

    def test_autonomous_target_rejects_whole_state_or_time_only_grounding(self):
        class UngroundedBackend:
            model_ref = "sha256:" + "7" * 64

            def generate(self, *, system, user, max_new_tokens):
                del system, user, max_new_tokens
                return json.dumps(
                    {
                        "commitment": {
                            "contract": "jenny.autonomous-commitment.v1",
                            "status": "ACTIVE",
                            "statement": "Invent a target without specific evidence.",
                            "rationale": "A whole-state hash exists.",
                            "evidence_keys": ["canonical-state", "temporal-now"],
                            "uncertainty": 0.9,
                        },
                        "actionability": "ACT_NOW",
                        "internal_request": "Invent a target without specific evidence.",
                        "rationale": "A whole-state hash exists.",
                        "predicted_observation": "Something may happen.",
                        "reasons_for": [],
                        "reasons_against": [],
                        "evidence_keys": ["canonical-state", "temporal-now"],
                        "uncertainty": 0.9,
                    }
                )

        with self.assertRaisesRegex(ValueError, "initiative evidence differs"):
            FrozenStructuredExperienceModel(
                UngroundedBackend()
            ).form_autonomous_target(
                cognitive_state={"situated_state": None},
                temporal={"trusted_utc": "2026-09-03T12:00:00Z"},
                memories=(),
                evidence_catalog={
                    "canonical-state": "sha256:" + "1" * 64,
                    "temporal-now": "sha256:" + "2" * 64,
                },
            )

    def test_situated_experience_receives_current_world_self_and_outcomes(self):
        backend = _Backend()
        cognitive_state = {
            "situated_state": {
                "world_model": ["A prior source returned contradictory evidence."],
                "self_model": ["The earlier prediction was too broad."],
                "focus": ["Test the narrowed prediction."],
            },
            "last_outcome": {"uncertainty": 0.4},
            "last_intent": None,
            "intent_ranking_evidence": [],
            "outcome_assessment_evidence": [
                {"prediction_assessments": [{"relation": "CONTRADICTED"}]}
            ],
            "qualitative_feedback": [],
            "self_appraisal_calibration": [],
        }

        result = FrozenStructuredExperienceModel(
            backend
        ).generate_situated_experience(
            "Test the revised expectation.",
            (),
            TemporalContext(3, None),
            cognitive_state=cognitive_state,
        )

        self.assertEqual(result.predicted_consequence, "Evidence should reduce uncertainty.")
        system, user, _ = backend.calls[0]
        self.assertIn("WORLD, SELF, FOCUS", system)
        self.assertIn("authored_artifacts", system)
        self.assertIn("do not promote its prose to observed fact", system)
        received = dict(json.loads(user)["current_cognitive_state"])
        manifest = received.pop("context_budget")
        self.assertEqual(received, cognitive_state)
        self.assertEqual(manifest["omitted"], [])  # everything fit; nothing dropped

    def test_observed_outcome_assessment_decodes_complete_grounded_state_proposal(self):
        expected = _valid_observed_outcome_assessment()
        backend = _ObservedOutcomeBackend((expected,))
        inputs = _observed_outcome_inputs()

        actual = FrozenStructuredExperienceModel(backend).assess_observed_outcome(
            **inputs
        )

        self.assertEqual(actual, expected)
        self.assertEqual(len(backend.calls), 1)
        system, user, maximum = backend.calls[0]
        self.assertEqual(maximum, 2_048)
        self.assertIn("Compare every supplied pre-action prediction", system)
        self.assertIn("Do not output numerical reward", system)
        supplied = json.loads(user)
        self.assertEqual(supplied["prediction_catalog"], inputs["prediction_catalog"])
        self.assertEqual(supplied["evidence_catalog"], inputs["evidence_catalog"])
        self.assertEqual(
            supplied["observable_consequence"], inputs["observable_consequence"]
        )
        self.assertEqual(
            {item["prediction_key"] for item in actual["prediction_assessments"]},
            set(inputs["prediction_catalog"]),
        )

    def test_authored_artifact_assessment_preserves_epistemic_boundary(self):
        expected = _valid_observed_outcome_assessment()
        backend = _ObservedOutcomeBackend((expected,))
        inputs = _observed_outcome_inputs()
        inputs["observable_consequence"]["observation"] = {
            "contract": "jenny.authored-artifact.observation.v1",
            "artifact_ref": "sha256:" + "9" * 64,
        }

        actual = FrozenStructuredExperienceModel(backend).assess_observed_outcome(
            **inputs
        )

        self.assertEqual(actual, expected)
        system = backend.calls[0][0]
        self.assertIn("private revisable model-authored work", system)
        self.assertIn("claims are not thereby verified", system)
        self.assertIn("not evidence of feeling, consciousness", system)

    def test_observed_outcome_assessment_repairs_unknown_evidence_once(self):
        invalid = _valid_observed_outcome_assessment()
        invalid["world_claims"][0]["evidence_keys"] = ["unknown-evidence"]
        repaired = _valid_observed_outcome_assessment()
        backend = _ObservedOutcomeBackend((invalid, repaired))

        actual = FrozenStructuredExperienceModel(backend).assess_observed_outcome(
            **_observed_outcome_inputs()
        )

        self.assertEqual(actual, repaired)
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("Repair the invalid assessment", backend.calls[1][0])
        repair_input = json.loads(backend.calls[1][1])
        self.assertEqual(repair_input["invalid_output"], json.dumps(invalid))
        self.assertEqual(
            repair_input["original_input"]["evidence_catalog"],
            _observed_outcome_inputs()["evidence_catalog"],
        )

    def test_observed_outcome_assessment_rejects_incomplete_prediction_coverage_after_one_repair(self):
        incomplete = _valid_observed_outcome_assessment()
        incomplete["prediction_assessments"] = incomplete[
            "prediction_assessments"
        ][:-1]
        backend = _ObservedOutcomeBackend((incomplete, incomplete))

        with self.assertRaisesRegex(ValueError, "prediction assessment count differs"):
            FrozenStructuredExperienceModel(backend).assess_observed_outcome(
                **_observed_outcome_inputs()
            )

        self.assertEqual(len(backend.calls), 2)
        self.assertIn("Repair the invalid assessment", backend.calls[1][0])

    def test_structured_experience_uses_one_model_driven_schema_repair(self):
        backend = _RepairingBackend()
        experience = FrozenStructuredExperienceModel(backend).generate_experience(
            "Remember one token.", (), TemporalContext(0, None)
        )
        self.assertEqual(experience.process_action, "VERIFY")
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("Repair one invalid", backend.calls[1][0])

    def test_loopback_openai_backend_is_revision_bound_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalOpenAICompatibleFrozenBackend(
                endpoint="http://127.0.0.1:30000/v1",
                served_model="jenny-qwen3.8-27b",
                model_path=Path(directory),
                revision="3" * 40,
                runtime_ref="sha256:" + "4" * 64,
            )
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse(
                    {"choices": [{"message": {"content": "bounded result"}}]}
                ),
            ) as call:
                self.assertEqual(
                    backend.generate(system="system", user="user", max_new_tokens=16),
                    "bounded result",
                )
            request = call.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "jenny-qwen3.8-27b")
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertTrue(backend.model_ref.startswith("sha256:"))
            self.assertTrue(payload["return_meta_info"])

    def test_loopback_openai_backend_sends_strict_json_schema(self):
        schema = {
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalOpenAICompatibleFrozenBackend(
                endpoint="http://127.0.0.1:30000/v1",
                served_model="jenny-qwen3.8-27b",
                model_path=Path(directory),
                revision="3" * 40,
                runtime_ref="sha256:" + "4" * 64,
            )
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": '{"decision":"answer"}'},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
            ) as call:
                actual = backend.generate_json(
                    system="system",
                    user="user",
                    max_new_tokens=64,
                    json_schema=schema,
                )

            self.assertEqual(actual, '{"decision":"answer"}')
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertEqual(
                payload["response_format"]["json_schema"]["schema"], schema
            )

    def test_openai_backend_records_content_free_usage_and_server_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalOpenAICompatibleFrozenBackend(
                endpoint="http://127.0.0.1:30000/v1",
                served_model="jenny-qwen3.8-27b",
                model_path=Path(directory),
                revision="3" * 40,
                runtime_ref="sha256:" + "4" * 64,
                trace_label="public-cortex-router",
            )
            response = {
                "id": "request-safe-id",
                "choices": [
                    {
                        "message": {"content": "private output"},
                        "finish_reason": "stop",
                        "meta_info": {
                            "e2e_latency": 2.0,
                            "queue_time": 0.1,
                            "decode_throughput": 9.5,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 20,
                    "total_tokens": 1220,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            }
            with capture_latency_trace(
                request_id="trace-test", source="HUMAN"
            ) as trace:
                with latency_phase("test.router"):
                    with patch(
                        "urllib.request.urlopen",
                        return_value=_HTTPResponse(response),
                    ):
                        self.assertEqual(
                            backend.generate(
                                system="secret system",
                                user="secret user",
                                max_new_tokens=32,
                            ),
                            "private output",
                        )
            self.assertEqual(trace["status"], "COMPLETED")
            call = trace["model_calls"][0]
            self.assertEqual(call["label"], "public-cortex-router")
            self.assertEqual(call["phase_path"], ["test.router"])
            self.assertEqual(call["prompt_tokens"], 1200)
            self.assertEqual(call["completion_tokens"], 20)
            self.assertEqual(call["cached_prompt_tokens"], 800)
            self.assertEqual(call["server_e2e_seconds"], 2.0)
            self.assertEqual(call["derived_decode_seconds"], 2.0)
            self.assertEqual(call["derived_time_to_first_token_seconds"], 0.0)
            encoded = json.dumps(trace)
            self.assertNotIn("secret system", encoded)
            self.assertNotIn("secret user", encoded)
            self.assertNotIn("private output", encoded)
            trace_ref, trace_path = persist_latency_trace(Path(directory), trace)
            self.assertTrue(trace_ref.startswith("sha256:"))
            self.assertEqual(json.loads(trace_path.read_text())["trace_ref"], trace_ref)

    def test_openai_backend_rejects_non_loopback_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "loopback"):
                LocalOpenAICompatibleFrozenBackend(
                    endpoint="https://example.com/v1",
                    served_model="model",
                    model_path=Path(directory),
                    revision="3" * 40,
                    runtime_ref="sha256:" + "4" * 64,
                )

    def test_openai_backend_can_bound_private_deliberation_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalOpenAICompatibleFrozenBackend(
                endpoint="http://127.0.0.1:30000/v1",
                served_model="jenny-qwen3.8-27b",
                model_path=Path(directory),
                revision="3" * 40,
                runtime_ref="sha256:" + "4" * 64,
                enable_thinking=True,
                reasoning_effort="low",
                maximum_output_tokens=2048,
            )
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse(
                    {"choices": [{"message": {"content": "checked result"}}]}
                ),
            ) as call:
                self.assertEqual(
                    backend.generate(system="system", user="user", max_new_tokens=768),
                    "checked result",
                )
            payload = json.loads(call.call_args.args[0].data)
            self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(payload["chat_template_kwargs"]["reasoning_effort"], "low")
            self.assertEqual(payload["max_tokens"], 768)

    def test_openai_backend_rejects_reasoning_effort_without_thinking(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires enable_thinking"):
                LocalOpenAICompatibleFrozenBackend(
                    endpoint="http://127.0.0.1:30000/v1",
                    served_model="jenny-qwen3.8-27b",
                    model_path=Path(directory),
                    revision="3" * 40,
                    runtime_ref="sha256:" + "4" * 64,
                    reasoning_effort="low",
                )

    def test_structured_experience_reflection_and_consolidation_are_separated(self):
        backend = _Backend()
        model = FrozenStructuredExperienceModel(backend)
        memories = (
            MemoryCandidate("sha256:" + "a" * 64, "Prior evidence.", 0.1),
        )
        experience = model.generate_experience(
            "Request", memories, TemporalContext(0, "sha256:" + "b" * 64)
        )
        self.assertEqual(experience.process_action, "VERIFY")
        receipt = FrozenQwenCortex(backend).execute("Request", experience, memories)
        consequence = ConsequenceVector(0.8, 0.7, 0.2, 0.5, 0.8, 0.6, 0.1, 1.0)
        reflection = model.reflect("Request", receipt, consequence, experience)
        consolidation = model.consolidate(
            "Request", experience, reflection, consequence
        )
        self.assertEqual(len(backend.calls), 4)
        self.assertIn("retained_principle", reflection)
        learned = json.loads(consolidation)
        self.assertEqual(learned["unfinished_patterns"], [])
        self.assertIn("observed consequence", learned["procedure"])
        self.assertIn("provenance", learned["limits"])
        self.assertIn("preserve both its name", backend.calls[3][0])
        self.assertIn("never replace the underlying rule", backend.calls[3][0])
        self.assertNotIn("objective_progress", backend.calls[0][1])

    def test_unevaluated_continuation_appraisal_is_explicitly_unverified(self):
        backend = _Backend()
        model = FrozenStructuredExperienceModel(backend)
        experience = model.generate_experience(
            "Inspect one bounded uncertainty.", (), TemporalContext(0, None)
        )
        receipt = FrozenQwenCortex(backend).execute(
            "Inspect one bounded uncertainty.", experience, ()
        )
        proposal = model.propose_continuation(
            "Inspect one bounded uncertainty.", receipt, experience
        )
        self.assertEqual(
            proposal["next_internal_request"], proposal["unfinished_patterns"][0]
        )
        self.assertTrue(proposal["self_appraisal"]["reasoned_judgment"].strip())
        self.assertEqual(len(proposal["world_model"]), 2)
        self.assertNotIn("overall_value", proposal["self_appraisal"])
        self.assertIn("No external result", proposal["self_appraisal"]["counterevidence"][0])
        self.assertIn("NOT been evaluated", backend.calls[-1][0])

    def test_capability_consolidation_targets_only_selected_revision(self):
        backend = _Backend()
        model = FrozenStructuredExperienceModel(backend)
        experience = StructuredExperience(
            "Interpret constraints.",
            "Compare the current case.",
            "Apply the selected procedure and verify.",
            "The receipt will test transfer.",
            ("Check the observable result.",),
            0.3,
        )
        consequence = ConsequenceVector(0.6, 0.7, 0.2, 0.5, 0.8, 0.6, 0.1, 1.0)
        active_ref = "sha256:" + "6" * 64
        value = json.loads(
            model.consolidate_with_capabilities(
                "Apply one bounded procedure.",
                experience,
                json.dumps({"analysis": "The receipt supports a bounded revision."}),
                consequence,
                (
                    {
                        "capability_key": "capability.synthetic-transfer",
                        "active_capability_ref": active_ref,
                        "revision_index": 2,
                        "procedure": "Use the prior process.",
                        "applicability": "Comparable cases.",
                        "limits": "One prior observation.",
                    },
                ),
            )
        )
        proposal = value["capability_proposal"]
        self.assertTrue(proposal["retain"])
        self.assertEqual(
            proposal["revision_target_key"], "capability.synthetic-transfer"
        )
        self.assertEqual(value["next_internal_request"], value["unfinished_patterns"][0])
        self.assertIn("not automatically a skill", backend.calls[-1][0])

    def test_situated_cortex_receives_trusted_now_and_memory_times(self):
        backend = _Backend()
        cortex = FrozenQwenCortex(backend)
        memory = MemoryCandidate(
            "sha256:" + "a" * 64,
            "Prior observed exchange.",
            0.1,
            acquired_ordinal=3,
            acquired_time_utc="2026-08-27T14:00:00Z",
            timezone="America/New_York",
        )
        experience = StructuredExperience(
            "Interpret relative time.", "VERIFY", "Compare trusted timestamps.",
            "A grounded relative phrase follows.", ("Check both timestamps.",), 0.1,
        )
        temporal_now = {
            "trusted_utc": "2026-09-02T18:00:00Z",
            "local_time": "2026-09-02T14:00:00-04:00",
            "local_timezone": "America/New_York",
        }
        capability = {
            "capability_evidence": [
                {
                    "epistemic_status": "MODEL_PROPOSAL_WITH_OBSERVED_CONSEQUENCE",
                    "procedure": "One bounded procedure.",
                    "limits": "One observed case only.",
                }
            ]
        }
        cortex.execute_with_context(
            "When was it?", experience, (memory,), temporal_now, capability
        )
        payload = json.loads(backend.calls[-1][1])
        self.assertEqual(payload["temporal_now"], temporal_now)
        received = dict(payload["cognitive_state"])
        manifest = received.pop("context_budget")
        self.assertEqual(received, capability)
        self.assertEqual(manifest["omitted"], [])  # everything fit; nothing dropped
        self.assertIn("authored_artifacts", backend.calls[-1][0])
        self.assertIn("never treat authorship as proof", backend.calls[-1][0])
        self.assertEqual(
            payload["retrieved_memories"][0]["acquired_time_utc"],
            "2026-08-27T14:00:00Z",
        )


if __name__ == "__main__":
    unittest.main()
