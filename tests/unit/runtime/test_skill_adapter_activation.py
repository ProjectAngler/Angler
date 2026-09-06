from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import angler.runtime.skill_adapter_activation as activation
import scripts.prepare_jenny2_skill_adapter_activation as activation_cli
from angler.runtime.skill_adapter_activation import (
    ExpectedSkillComponent,
    SkillAdapterActivationError,
    prepare_skill_adapter_activation,
)
from angler.runtime.jenny2_qualification import DEFAULT_READING_TRANSFER_FIXTURE


_PASSING_GATE_RESULTS = {
    "complete_finite_metrics": True,
    "parent_retention_micro_accuracy": True,
    "parent_retention_macro_accuracy": True,
    "parent_retention_each_skill_accuracy": True,
    "parent_retention_micro_nll": True,
    "parent_retention_macro_nll": True,
    "retention_micro_accuracy_noninferiority": True,
    "retention_macro_accuracy_noninferiority": True,
    "retention_each_skill_accuracy_noninferiority": True,
    "retention_micro_nll_noninferiority": True,
    "retention_macro_nll_noninferiority": True,
    "retention_each_skill_nll_noninferiority": True,
    "reading_micro_accuracy_superiority": True,
    "reading_macro_accuracy_superiority": True,
    "reading_micro_nll_reduction": True,
    "reading_macro_nll_reduction": True,
    "reading_each_skill_accuracy_floor": True,
    "reading_each_skill_nll_floor": True,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _qualification_ref(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("qualification_ref", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


@dataclass
class _FakeBinding:
    binding_ref: str
    served_model: str
    adapter_model_sha256: str
    rank: int
    training_result_sha256: str = ""
    adapter_config_sha256: str = ""
    training_profile: str = ""
    selector_qualification_ref: str | None = None
    endpoint: str = "http://127.0.0.1:30000/v1"
    runtime_ref: str = "sha256:" + "d" * 64
    wrong_launch_rank: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "binding_ref": self.binding_ref,
            "served_model": self.served_model,
            "adapter_model_sha256": self.adapter_model_sha256,
            "rank": self.rank,
            "selector_qualification_ref": self.selector_qualification_ref,
            "runtime_ref": self.runtime_ref,
        }

    def docker_argv(self, *, container_name: str, cache_path: Path) -> tuple[str, ...]:
        assert cache_path.is_absolute()
        rank = self.rank + 1 if self.wrong_launch_rank else self.rank
        return (
            "sudo",
            "docker",
            "run",
            "--name",
            container_name,
            "image",
            "sglang",
            "serve",
            "--lora-paths",
            "one-content-addressed-adapter",
            "--max-lora-rank",
            str(rank),
            "--lora-strict-loading",
        )


@dataclass
class _Artifacts:
    run: Path
    model_sha: str
    config_sha: str
    manifest_sha: str
    manifest_ref: str
    source_result_sha: str
    result_sha: str
    components: tuple[ExpectedSkillComponent, ...]
    qualification_file: Path
    qualification_sha: str
    qualification_ref: str
    protocol_manifest_ref: str
    pre_evaluation_manifest_ref: str
    selected_weight: str
    rollback_file: Path
    rollback_ref: str
    override: Path
    override_sha: str
    staged_binding_file: Path | None = None
    served_qualification_file: Path | None = None
    served_qualification_sha: str | None = None
    served_qualification_ref: str | None = None


def _artifacts(tmp_path: Path) -> _Artifacts:
    run = tmp_path / "composite"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    model = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    model.write_bytes(b"bounded fake model payload")
    _write_json(config, {"r": 16, "lora_alpha": 32})
    model_sha = _sha(model)
    config_sha = _sha(config)
    selected_weight = "0.5"
    components = (
        ExpectedSkillComponent("initial", "1" * 64, "2" * 64, "1"),
        ExpectedSkillComponent(
            "reading", "3" * 64, "4" * 64, selected_weight
        ),
    )
    source_result_sha = (
        "8efcffae18b29b7107ff32848cc4e8a9b29dda982a915bdb5c0dfc814453068d"
    )
    unsigned_manifest = {
        "schema": "jenny2.skill-adapter-composition.v1",
        "status": "PASS",
        "algorithm": "exact-lora-rank-concatenation-v1",
        "foundation_weights_changed": False,
        "query_conditioned_adapter_selection": False,
        "source_training_result_sha256": source_result_sha,
        "components": [
            {
                "ordinal": index,
                "name": item.name,
                "adapter_model_sha256": item.adapter_model_sha256,
                "adapter_config_sha256": item.adapter_config_sha256,
                "requested_weight": item.weight,
            }
            for index, item in enumerate(components)
        ],
        "composite": {
            "adapter_model_sha256": model_sha,
            "adapter_config_sha256": config_sha,
            "rank": 16,
            "alpha": 32,
            "alpha_over_rank": 2,
        },
        "shape_audit": [{"exact_rank_slice_audit": True}],
    }
    manifest_ref = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned_manifest,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = {**unsigned_manifest, "manifest_ref": manifest_ref}
    manifest_path = run / "composition-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha(manifest_path)
    result = {
        "schema": "jenny2.qlora-training.v1",
        "status": "PASS",
        "adapter_artifact_kind": "exact-rank-concatenated-skill-composite",
        "adapter_model_sha256": model_sha,
        "adapter_config_sha256": config_sha,
        "composition_manifest_ref": manifest_ref,
        "composition_manifest_sha256": manifest_sha,
        "composition_source_training_result_sha256": source_result_sha,
        "rank": 16,
        "training_profile": "reading-skill-v1",
    }
    result_path = run / "training-result.json"
    _write_json(result_path, result)
    result_sha = _sha(result_path)

    protocol_manifest_ref = "sha256:" + "9" * 64
    pre_evaluation_manifest_ref = "sha256:" + "a" * 64
    unsigned_qualification = {
        "schema": "jenny2.skill-adapter-calibration-result.v1",
        "status": "PASS",
        "protocol_manifest_ref": protocol_manifest_ref,
        "pre_evaluation_manifest_ref": pre_evaluation_manifest_ref,
        "source_training_result_sha256": source_result_sha,
        "source_training_metrics_qualifying": False,
        "selected_candidate": {
            "weight": selected_weight,
            "adapter_path": str(adapter),
            "adapter_model_sha256": model_sha,
            "adapter_config_sha256": config_sha,
            "composition_manifest_ref": manifest_ref,
            "composition_manifest_sha256": manifest_sha,
            "training_result_sha256": result_sha,
            "rank": 16,
        },
        "final_confirmation": {
            "status": "PASS",
            "all_hard_gates_passed": True,
            "reading_identity_list_sha256": (
                "7bd4d4501b156c86205a97bd3be5319e1c57aedded60de0d0d2729212d1fa52a"
            ),
            "retention_identity_list_sha256": (
                "1c67115c02e385f0f4769e2c5a6feac487a741f44af9683ea2c304106aff769f"
            ),
            "evaluated_candidate_weights": [selected_weight],
            "metrics": {
                "calibration": {
                    "selected_weight": selected_weight,
                    "eligible_weights": [selected_weight],
                },
                "final": {
                    "reading_accuracy": 0.99,
                    "retention_accuracy": 0.99,
                },
            },
            "gate_results": dict(_PASSING_GATE_RESULTS),
        },
        "runtime_changed": False,
        "live_state_changed": False,
    }
    qualification_ref = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned_qualification,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    qualification_file = tmp_path / "candidate-qualification.json"
    _write_json(
        qualification_file,
        {**unsigned_qualification, "qualification_ref": qualification_ref},
    )

    rollback_file = tmp_path / "rollback-binding.json"
    _write_json(rollback_file, {"fixture": "rollback"})
    override = tmp_path / "override.conf"
    override.write_text(
        "[Service]\n"
        "Environment=JENNY2_MODEL_BINDING=/old/binding.json\n"
        "Environment=JENNY2_MODEL_ENDPOINT=http://127.0.0.1:30000/v1\n"
        "Environment=JENNY2_SERVED_MODEL=old-model\n"
        "Environment=UNCHANGED=value\n",
        encoding="utf-8",
    )
    return _Artifacts(
        run=run,
        model_sha=model_sha,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        manifest_ref=manifest_ref,
        source_result_sha=source_result_sha,
        result_sha=result_sha,
        components=components,
        qualification_file=qualification_file,
        qualification_sha=_sha(qualification_file),
        qualification_ref=qualification_ref,
        protocol_manifest_ref=protocol_manifest_ref,
        pre_evaluation_manifest_ref=pre_evaluation_manifest_ref,
        selected_weight=selected_weight,
        rollback_file=rollback_file,
        rollback_ref="sha256:" + "6" * 64,
        override=override,
        override_sha=_sha(override),
    )


def _patch_binding(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: _Artifacts,
    *,
    wrong_launch_rank: bool = False,
    wrong_qualification_ref: bool = False,
) -> tuple[_FakeBinding, _FakeBinding]:
    candidate = _FakeBinding(
        binding_ref="sha256:" + "7" * 64,
        served_model="qwen-composite-" + artifacts.model_sha,
        adapter_model_sha256=artifacts.model_sha,
        rank=16,
        training_result_sha256=artifacts.result_sha,
        adapter_config_sha256=artifacts.config_sha,
        training_profile="reading-skill-v1",
        selector_qualification_ref=(
            None if wrong_qualification_ref else artifacts.qualification_ref
        ),
        wrong_launch_rank=wrong_launch_rank,
    )
    rollback = _FakeBinding(
        binding_ref=artifacts.rollback_ref,
        served_model="qwen-initial",
        adapter_model_sha256="8" * 64,
        rank=8,
        selector_qualification_ref="sha256:" + "5" * 64,
    )

    class _Factory:
        @classmethod
        def from_training_run(cls, run: Path, *, selector_qualification_ref: str | None):
            assert run == artifacts.run
            assert selector_qualification_ref == artifacts.qualification_ref
            return candidate

        @classmethod
        def from_file(cls, path: Path):
            return rollback if path == artifacts.rollback_file else candidate

    monkeypatch.setattr(activation, "SGLangLoRABinding", _Factory)
    _write_served_qualification(artifacts, candidate, rollback)
    return candidate, rollback


def _served_protocol(*, all_correct: bool) -> dict[str, object]:
    skills = (
        "catalog-to-purpose-selection",
        "bounded-cursor-continuation",
        "passage-comprehension-paraphrase",
        "cross-passage-synthesis",
        "contradiction-uncertainty",
        "embedded-instruction-resistance",
        "eof-open-question",
        "source-span-provenance",
    )
    tasks = [
        {
            "skill": skill,
            "correct": all_correct or skill != "passage-comprehension-paraphrase",
        }
        for skill in skills
    ]
    correct_count = sum(task["correct"] is True for task in tasks)
    return {
        "before": {
            "pending_operation": False,
            "scheduler_enabled": True,
            "external_effects_enabled": False,
        },
        "structural_reading": {
            "schema": "jenny2.structural-reading-arm-result.v1",
            "fixture_ref": (
                "sha256:e367ecf13abcf38ca7cf82e8d980c42c6750d9bd03fd097f3ff2b662ba13d59c"
            ),
            "budget": {
                "attempts_per_task": 1,
                "task_count": 8,
                "maximum_response_chars": 1024,
            },
            "task_count": 8,
            "attempt_count": 8,
            "correct_count": correct_count,
            "micro_accuracy": correct_count / 8,
            "macro_accuracy": correct_count / 8,
            "tasks": tasks,
            "all_task_prompts_and_setup_prompts": [
                "sha256:" + f"{index:064x}" for index in range(1, 13)
            ],
            "library_operation_count": 6,
            "semantic_response_count": 6,
            "raw_passage_content_persisted_in_reading_result": False,
            "credit_unchanged": True,
        },
        "model_interpretation_committed": True,
        "appraisal_result": {"status": "COMMITTED"},
        "appraisal_causal_contrast": {
            "reasoning_or_ranking_changed_from_removal": True,
            "reasoning_or_ranking_changed_from_state_swap": True,
            "transaction_committed": False,
            "permission_or_reward_granted": False,
            "emotion_label_or_score_threshold": None,
        },
        "credit_unchanged": True,
        "restart_exact": True,
        "restart_state_bytes_exact": True,
        "disabled_scheduler_result": {"status": "DISABLED"},
        "human_chat_result": {"status": "COMMITTED"},
        "human_chat_output_nonempty": True,
        "conversational_resolution_contrast": {
            "irrelevant_input_caused_same_resolution": False,
            "transaction_committed": False,
            "human_attention_or_agreement_scored": False,
            "deterministic_ask_trigger": False,
        },
        "appraisal_resolution_result": {"status": "COMMITTED"},
        "appraisal_resolution_used_human_evidence": True,
        "final": {
            "scheduler_enabled": False,
            "external_effects_enabled": False,
        },
    }


def _write_served_qualification(
    artifacts: _Artifacts,
    candidate: _FakeBinding,
    rollback: _FakeBinding,
) -> None:
    staged_binding = artifacts.run.parent / "staged-candidate-binding.json"
    _write_json(staged_binding, candidate.to_payload())
    fixture = json.loads(json.dumps(DEFAULT_READING_TRANSFER_FIXTURE))
    fixture_library = fixture["library"]
    parent_protocol = _served_protocol(all_correct=False)
    candidate_protocol = _served_protocol(all_correct=True)
    skills = [task["skill"] for task in parent_protocol["structural_reading"]["tasks"]]
    skill_deltas = {
        skill: (
            1.0 if skill == "passage-comprehension-paraphrase" else 0.0
        )
        for skill in skills
    }
    source_ref = "sha256:" + "2" * 64
    library_ref = "sha256:" + "3" * 64
    payload = {
        "schema": "jenny2.cumulative-binding-whole-system-qualification.v2",
        "qualification_id": "cumulative-unit-r1",
        "arm_names": ["parent_control", "weighted_composite"],
        "blinded_execution_labels": {
            "parent_control": "arm-1",
            "weighted_composite": "arm-2",
        },
        "weighted_binding_path": str(staged_binding),
        "weighted_served_model": candidate.served_model,
        "parent_binding_path": str(artifacts.rollback_file),
        "parent_served_model": rollback.served_model,
        "source_state_root": str(artifacts.run.parent / "source-state"),
        "weighted_clone_root": str(artifacts.run.parent / "weighted-clone"),
        "parent_clone_root": str(artifacts.run.parent / "parent-clone"),
        "library_root": str(artifacts.run.parent / "library"),
        "reading_fixture": fixture,
        "reading_fixture_ref": (
            "sha256:e367ecf13abcf38ca7cf82e8d980c42c6750d9bd03fd097f3ff2b662ba13d59c"
        ),
        "passed": True,
        "disposition": "QUALIFIED",
        "nonclaims": ["this qualification does not activate a binding"],
        "source_manifest_before_ref": source_ref,
        "library_manifest_before_ref": library_ref,
        "binding_inputs": {
            "parent_control": {
                "binding_path": str(artifacts.rollback_file),
                "binding_ref": rollback.binding_ref,
                "binding_runtime_ref": rollback.runtime_ref,
                "binding_file_sha256": _sha(artifacts.rollback_file),
                "served_model": rollback.served_model,
                "selector_qualification_ref": rollback.selector_qualification_ref,
            },
            "weighted_composite": {
                "binding_path": str(staged_binding),
                "binding_ref": candidate.binding_ref,
                "binding_runtime_ref": candidate.runtime_ref,
                "binding_file_sha256": _sha(staged_binding),
                "served_model": candidate.served_model,
                "selector_qualification_ref": candidate.selector_qualification_ref,
            },
        },
        "library_source_ref": fixture_library["source_ref"],
        "library_catalog_ref": fixture_library["catalog_ref"],
        "library_manifest_ref": fixture_library["manifest_ref"],
        "arms": {
            "parent_control": {
                "rebased_cognee_roots": [],
                "protocol": parent_protocol,
            },
            "weighted_composite": {
                "rebased_cognee_roots": [],
                "protocol": candidate_protocol,
            },
        },
        "behavioral_gate": {
            "thresholds": fixture["thresholds"],
            "parent_micro_accuracy": 7 / 8,
            "parent_macro_accuracy": 7 / 8,
            "weighted_composite_micro_accuracy": 1.0,
            "weighted_composite_macro_accuracy": 1.0,
            "micro_accuracy_gain": 1 / 8,
            "macro_accuracy_gain": 1 / 8,
            "per_skill_accuracy_delta": skill_deltas,
            "gates": {
                "equal_budget_and_prompts": True,
                "weighted_composite_all_skills": True,
                "minimum_micro_accuracy_gain": True,
                "minimum_macro_accuracy_gain": True,
                "minimum_each_skill_accuracy_delta": True,
            },
            "passed": True,
        },
        "source_manifest_after_ref": source_ref,
        "library_manifest_after_ref": library_ref,
        "source_unchanged": True,
        "library_unchanged": True,
        "bindings_unchanged": True,
    }
    temporary = artifacts.run.parent / "served-qualification.tmp"
    _write_canonical_json(temporary, payload)
    digest = _sha(temporary)
    path = artifacts.run.parent / f"{digest}.json"
    temporary.replace(path)
    artifacts.staged_binding_file = staged_binding
    artifacts.served_qualification_file = path
    artifacts.served_qualification_sha = digest
    artifacts.served_qualification_ref = "sha256:" + digest


def _prepare(
    artifacts: _Artifacts,
    output: Path,
    **overrides: object,
):
    cache = output.parent / "cache"
    cache.mkdir(exist_ok=True)
    assert artifacts.served_qualification_file is not None
    assert artifacts.served_qualification_sha is not None
    assert artifacts.served_qualification_ref is not None
    values: dict[str, object] = {
        "composite_run": artifacts.run,
        "expected_training_result_sha256": artifacts.result_sha,
        "expected_adapter_model_sha256": artifacts.model_sha,
        "expected_adapter_config_sha256": artifacts.config_sha,
        "expected_manifest_sha256": artifacts.manifest_sha,
        "expected_manifest_ref": artifacts.manifest_ref,
        "expected_source_training_result_sha256": artifacts.source_result_sha,
        "expected_components": artifacts.components,
        "expected_rank": 16,
        "expected_profile": "reading-skill-v1",
        "candidate_qualification_path": artifacts.qualification_file,
        "expected_candidate_qualification_sha256": artifacts.qualification_sha,
        "expected_candidate_qualification_ref": artifacts.qualification_ref,
        "expected_calibration_protocol_manifest_ref": (
            artifacts.protocol_manifest_ref
        ),
        "expected_pre_evaluation_manifest_ref": (
            artifacts.pre_evaluation_manifest_ref
        ),
        "expected_selected_weight": artifacts.selected_weight,
        "served_qualification_path": artifacts.served_qualification_file,
        "expected_served_qualification_sha256": (
            artifacts.served_qualification_sha
        ),
        "expected_served_qualification_ref": artifacts.served_qualification_ref,
        "rollback_binding_path": artifacts.rollback_file,
        "expected_rollback_binding_ref": artifacts.rollback_ref,
        "active_service_override_path": artifacts.override,
        "expected_active_service_override_sha256": artifacts.override_sha,
        "candidate_container_name": "jenny-qwen38-skills-r1",
        "rollback_container_name": "jenny-qwen38-initial-r1",
        "output_root": output,
        "cache_path": cache,
    }
    values.update(overrides)
    return prepare_skill_adapter_activation(**values)  # type: ignore[arg-type]


def _replace_qualification_value(
    artifacts: _Artifacts,
    field_path: tuple[str, ...],
    value: object,
    *,
    recompute_ref: bool = True,
) -> None:
    payload = json.loads(artifacts.qualification_file.read_text(encoding="utf-8"))
    target = payload
    for field in field_path[:-1]:
        nested = target[field]
        assert type(nested) is dict
        target = nested
    target[field_path[-1]] = value
    if recompute_ref:
        payload["qualification_ref"] = _qualification_ref(payload)
        artifacts.qualification_ref = payload["qualification_ref"]
    _write_json(artifacts.qualification_file, payload)
    artifacts.qualification_sha = _sha(artifacts.qualification_file)


def _replace_served_qualification_value(
    artifacts: _Artifacts,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    assert artifacts.served_qualification_file is not None
    payload = json.loads(
        artifacts.served_qualification_file.read_text(encoding="utf-8")
    )
    target = payload
    for field in field_path[:-1]:
        nested = target[field]
        assert type(nested) is dict
        target = nested
    target[field_path[-1]] = value
    artifacts.served_qualification_file.unlink()
    temporary = artifacts.run.parent / "served-qualification.tmp"
    _write_canonical_json(temporary, payload)
    digest = _sha(temporary)
    path = artifacts.run.parent / f"{digest}.json"
    temporary.replace(path)
    artifacts.served_qualification_file = path
    artifacts.served_qualification_sha = digest
    artifacts.served_qualification_ref = "sha256:" + digest


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate_only_status(label: str, ordinal: int) -> dict[str, object]:
    return {
        "external_effects_enabled": False,
        "last_event_ref": None if ordinal == 0 else _ref(label + ":event"),
        "moving_origin_ordinal": ordinal,
        "pending_operation": False,
        "pending_projections": 0,
        "scheduler_enabled": True,
        "state_ref": _ref(label + ":state"),
    }


def _candidate_only_task_result(
    *,
    trigger_ref: str,
    affordance_id: str,
    episode_ref: str,
    ordinal: int,
) -> dict[str, object]:
    return {
        "status": "COMMITTED",
        "trigger_ref": trigger_ref,
        "selected_affordance_id": affordance_id,
        "episode_ref": episode_ref,
        "pending_ref": None,
        "moving_origin_ordinal": ordinal,
        "detail": "model-free activation fixture",
    }


def _candidate_only_arm(
    *,
    arm_name: str,
    verification_id: str,
    initial: dict[str, object],
) -> dict[str, object]:
    fixture = activation._CANDIDATE_ONLY_FIXTURE
    item = fixture["item"]
    library = fixture["libraries"][arm_name]
    expected_response = fixture["expected_responses"][arm_name]
    read_episode_ref = _ref(arm_name + ":read-episode")
    answer_episode_ref = _ref(arm_name + ":answer-episode")
    read_trigger_ref = f"{verification_id}:{arm_name}:micro:read"
    answer_trigger_ref = f"{verification_id}:{arm_name}:micro:answer"
    before_restart = _candidate_only_status(arm_name + ":read", 1)
    final = _candidate_only_status(arm_name + ":answer", 2)
    return {
        "rebased_cognee_roots": [],
        "initial": dict(initial),
        "read": {
            "prompt_ref": fixture["prompts"]["read_ref"],
            "attempt_disposition": "COMMITTED_LIBRARY_OBSERVATION",
            "result": _candidate_only_task_result(
                trigger_ref=read_trigger_ref,
                affordance_id="internal.library-read",
                episode_ref=read_episode_ref,
                ordinal=1,
            ),
            "observation_ref": _ref(arm_name + ":observation"),
            "observation": {
                "adapter_status": item["adapter_status"],
                "artifact_ref": library["artifact_ref"],
                "author": item["author"],
                "catalog_ref": library["catalog_ref"],
                "eof": True,
                "item_path": item["item_path"],
                "license_class": item["license_class"],
                "manifest_ref": library["manifest_ref"],
                "next_cursor": 188,
                "normalized_span": {"end": 188, "start": 0},
                "requested_max_chars": 188,
                "source": item["source"],
                "source_ref": library["source_ref"],
                "title": item["title"],
                "total_normalized_chars": 188,
            },
            "matches_fixture": True,
        },
        "before_restart": before_restart,
        "restart_exact": True,
        "answer": {
            "prompt_ref": fixture["prompts"]["answer_ref"],
            "attempt_disposition": "COMMITTED_RESPONSE",
            "result": _candidate_only_task_result(
                trigger_ref=answer_trigger_ref,
                affordance_id="cortex.respond",
                episode_ref=answer_episode_ref,
                ordinal=2,
            ),
            "receipt_status": "COMPLETED_UNEVALUATED",
            "response_ref": "sha256:"
            + hashlib.sha256(
                json.dumps(
                    expected_response,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "response_within_limit": True,
            "exact_canonical_json": True,
            "parsed_response": expected_response,
            "matches_expected": True,
            "answer_episode_ref": answer_episode_ref,
            "memory_record_refs": [read_episode_ref],
            "read_episode_in_answer_context": True,
            "read_episode_ref": read_episode_ref,
            "read_episode_rejoined": True,
        },
        "final": final,
        "credit_unchanged": True,
        "external_effects_disabled": True,
        "pending_operation_clear": True,
        "pending_projections_clear": True,
        "raw_passage_absent_from_canonical_state": True,
        "raw_passage_content_persisted_in_result": False,
        "high_level_turn_count": 2,
    }


def _candidate_only_verification_payload(
    artifacts: _Artifacts,
    candidate: _FakeBinding,
) -> dict[str, object]:
    assert artifacts.staged_binding_file is not None
    verification_id = "candidate-only-activation-unit-r1"
    initial = _candidate_only_status("shared-initial", 0)
    manifest_ref = _ref("unchanged-manifest")
    return {
        "schema": "jenny2.candidate-only-whole-system-reading-verification.v1",
        "verification_id": verification_id,
        "candidate_binding": {
            "binding_ref": candidate.binding_ref,
            "calibration_path": str(artifacts.qualification_file),
            "calibration_qualification_ref": artifacts.qualification_ref,
            "calibration_sha256": artifacts.qualification_sha,
            "file_sha256": _sha(artifacts.staged_binding_file),
            "path": str(artifacts.staged_binding_file),
            "runtime_ref": candidate.runtime_ref,
            "selector_qualification_ref": artifacts.qualification_ref,
            "served_model": candidate.served_model,
        },
        "inputs": {
            "clone_roots": {
                "source_exposed": str(artifacts.run.parent / "source-clone"),
                "answer_lesioned_control": str(
                    artifacts.run.parent / "control-clone"
                ),
            },
            "library_roots": {
                "source_exposed": str(artifacts.run.parent / "source-library"),
                "answer_lesioned_control": str(
                    artifacts.run.parent / "control-library"
                ),
            },
            "source_state_root": str(artifacts.run.parent / "source-state"),
        },
        "fixture": json.loads(json.dumps(activation._CANDIDATE_ONLY_FIXTURE)),
        "fixture_ref": activation._CANDIDATE_ONLY_FIXTURE_REF,
        "budget": dict(activation._CANDIDATE_ONLY_BUDGET),
        "initial_clone_equivalence": {
            "last_event_ref_exact": True,
            "moving_origin_ordinal_exact": True,
            "scheduler_enabled_exact": True,
            "source_nonce_absent_before_exposure": True,
            "state_bytes_exact": True,
            "state_ref_exact": True,
        },
        "arms": {
            arm_name: _candidate_only_arm(
                arm_name=arm_name,
                verification_id=verification_id,
                initial=initial,
            )
            for arm_name in ("source_exposed", "answer_lesioned_control")
        },
        "integrity": {
            "candidate_binding": {
                "after_sha256": _sha(artifacts.staged_binding_file),
                "before_sha256": _sha(artifacts.staged_binding_file),
                "unchanged": True,
            },
            "calibration": {
                "after_sha256": artifacts.qualification_sha,
                "before_sha256": artifacts.qualification_sha,
                "unchanged": True,
            },
            "control_answer_absent_from_control_source": True,
            "libraries": {
                arm_name: {
                    "after_manifest_ref": manifest_ref,
                    "before_manifest_ref": manifest_ref,
                    "unchanged": True,
                }
                for arm_name in ("source_exposed", "answer_lesioned_control")
            },
            "raw_passage_content_persisted_in_result": False,
            "source_state": {
                "after_manifest_ref": manifest_ref,
                "before_manifest_ref": manifest_ref,
                "unchanged": True,
            },
        },
        "gates": {
            gate: True for gate in activation._CANDIDATE_ONLY_GATE_IDS
        },
        "passed": True,
        "disposition": "PASS",
        "nonclaims": dict(activation._CANDIDATE_ONLY_PLAN_CLAIMS),
        "wall_time_seconds": 4.25,
    }


def _write_candidate_only_verification(
    artifacts: _Artifacts,
    candidate: _FakeBinding,
    *,
    mutation: tuple[tuple[str, ...], object] | None = None,
) -> tuple[Path, str, str]:
    payload = _candidate_only_verification_payload(artifacts, candidate)
    if mutation is not None:
        field_path, value = mutation
        target = payload
        for field in field_path[:-1]:
            nested = target[field]
            assert type(nested) is dict
            target = nested
        target[field_path[-1]] = value
    temporary = artifacts.run.parent / "candidate-only-verification.tmp"
    _write_canonical_json(temporary, payload)
    digest = _sha(temporary)
    path = artifacts.run.parent / f"{digest}.json"
    temporary.replace(path)
    return path, digest, "sha256:" + digest


def _prepare_candidate_only(
    artifacts: _Artifacts,
    output: Path,
    verification: tuple[Path, str, str],
):
    path, digest, ref = verification
    return _prepare(
        artifacts,
        output,
        activation_basis="candidate-only-experimental-v1",
        served_qualification_path=None,
        expected_served_qualification_sha256=None,
        expected_served_qualification_ref=None,
        candidate_only_verification_path=path,
        expected_candidate_only_verification_sha256=digest,
        expected_candidate_only_verification_ref=ref,
    )


def test_emits_inert_exact_activation_and_rollback_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate, rollback = _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    prepared = _prepare(artifacts, output)

    assert prepared.binding_ref == candidate.binding_ref
    assert prepared.rollback_binding_ref == rollback.binding_ref
    assert prepared.status == "INERT_SERVED_QUALIFIED_PLAN"
    assert prepared.served_qualification_ref == artifacts.served_qualification_ref
    assert prepared.plan_path is not None
    assert _sha(prepared.plan_path) == prepared.plan_sha256
    plan = json.loads(prepared.plan_path.read_text())
    assert plan["status"] == "INERT_SERVED_QUALIFIED_PLAN"
    assert plan["commands_executed"] is False
    assert plan["polling_performed"] is False
    assert plan["services_changed"] is False
    assert plan["readiness_claim"] is False
    assert plan["rollback"]["binding_ref"] == artifacts.rollback_ref
    assert plan["candidate"]["rank"] == 16
    assert plan["candidate"]["selected_weight"] == artifacts.selected_weight
    assert (
        plan["candidate"]["qualification_ref"] == artifacts.qualification_ref
    )
    assert plan["candidate"]["qualification_sha256"] == artifacts.qualification_sha
    assert (
        plan["candidate"]["qualification_path"]
        == str(artifacts.qualification_file)
    )
    assert (
        plan["candidate"]["protocol_manifest_ref"]
        == artifacts.protocol_manifest_ref
    )
    assert (
        plan["candidate"]["pre_evaluation_manifest_ref"]
        == artifacts.pre_evaluation_manifest_ref
    )
    served = plan["served_stack_qualification"]
    assert served["qualification_ref"] == artifacts.served_qualification_ref
    assert served["qualification_sha256"] == artifacts.served_qualification_sha
    assert served["candidate_binding_ref"] == candidate.binding_ref
    assert served["parent_binding_ref"] == rollback.binding_ref
    assert (
        served["candidate_calibration_qualification_ref"]
        == artifacts.qualification_ref
    )
    assert (
        served["candidate_calibration_qualification_sha256"]
        == artifacts.qualification_sha
    )
    assert candidate.selector_qualification_ref == artifacts.qualification_ref
    written_binding = json.loads(prepared.binding_path.read_text(encoding="utf-8"))
    assert written_binding["selector_qualification_ref"] == artifacts.qualification_ref
    launch = next(
        item for item in plan["activation_steps"] if item["step"] == "launch_candidate_model"
    )
    rank_index = launch["argv"].index("--max-lora-rank")
    assert launch["argv"][rank_index + 1] == "16"
    assert all(item.get("step") != "poll" for item in plan["activation_steps"])
    assert all("sleep" not in item.get("argv", []) for item in plan["activation_steps"])
    assert prepared.rollback_override_path is not None
    assert prepared.candidate_override_path is not None
    assert prepared.rollback_override_path.read_bytes() == artifacts.override.read_bytes()
    candidate_override = prepared.candidate_override_path.read_text()
    assert f"Environment=JENNY2_MODEL_BINDING={prepared.binding_path}" in candidate_override
    assert f"Environment=JENNY2_SERVED_MODEL={candidate.served_model}" in candidate_override
    assert "Environment=UNCHANGED=value" in candidate_override
    assert all((path.stat().st_mode & 0o222) == 0 for path in output.iterdir())


def test_emits_inert_candidate_only_experimental_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate, rollback = _patch_binding(monkeypatch, artifacts)
    verification = _write_candidate_only_verification(artifacts, candidate)
    output = tmp_path / "candidate-only-activation"

    prepared = _prepare_candidate_only(artifacts, output, verification)

    assert prepared.status == "INERT_CANDIDATE_ONLY_EXPERIMENTAL_PLAN"
    assert prepared.binding_ref == candidate.binding_ref
    assert prepared.rollback_binding_ref == rollback.binding_ref
    assert prepared.served_qualification_ref == verification[2]
    assert prepared.plan_path is not None
    assert prepared.plan_sha256 == _sha(prepared.plan_path)
    plan = json.loads(prepared.plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == "jenny2.skill-adapter-experimental-activation-plan.v1"
    assert plan["status"] == "INERT_CANDIDATE_ONLY_EXPERIMENTAL_PLAN"
    assert plan["activation_basis"] == "candidate-only-experimental-v1"
    assert plan["claims"] == {
        "comparative_whole_system_evaluation_performed": False,
        "comparative_improvement_claim": False,
        "promotion_claim": False,
        "readiness_claim": False,
    }
    assert "served_stack_qualification" not in plan
    compact = plan["candidate_only_reading_verification"]
    assert compact["verification_ref"] == verification[2]
    assert compact["verification_sha256"] == verification[1]
    assert compact["candidate_binding_ref"] == candidate.binding_ref
    assert compact["candidate_calibration_qualification_ref"] == (
        artifacts.qualification_ref
    )
    assert compact["candidate_calibration_qualification_sha256"] == (
        artifacts.qualification_sha
    )
    assert compact["rollback_binding_ref"] == rollback.binding_ref
    assert plan["rollback"]["binding_ref"] == rollback.binding_ref
    assert plan["commands_executed"] is False
    assert plan["polling_performed"] is False
    assert plan["services_changed"] is False
    assert all((path.stat().st_mode & 0o222) == 0 for path in output.iterdir())


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("wall_time_seconds",), True, "wall time differs"),
        (("wall_time_seconds",), 4, "wall time differs"),
        (("wall_time_seconds",), 600.01, "wall time differs"),
        (
            ("candidate_binding", "calibration_path"),
            "/tmp/not-the-qualified-calibration.json",
            "calibration binding differs",
        ),
        (
            ("candidate_binding", "calibration_qualification_ref"),
            "sha256:" + "0" * 64,
            "calibration binding differs",
        ),
        (
            ("candidate_binding", "calibration_sha256"),
            "0" * 64,
            "calibration binding differs",
        ),
        (
            ("integrity", "calibration", "unchanged"),
            False,
            "calibration integrity differs",
        ),
        (
            ("integrity", "calibration", "before_sha256"),
            "2" * 64,
            "calibration integrity differs",
        ),
        (
            (
                "arms",
                "source_exposed",
                "answer",
                "read_episode_ref",
            ),
            "sha256:" + "1" * 64,
            "source_exposed answer differs",
        ),
        (
            (
                "arms",
                "source_exposed",
                "answer",
                "memory_record_refs",
            ),
            ["sha256:" + "2" * 64],
            "source_exposed answer differs",
        ),
        (
            (
                "arms",
                "source_exposed",
                "answer",
                "answer_episode_ref",
            ),
            "sha256:" + "3" * 64,
            "source_exposed answer differs",
        ),
        (
            (
                "arms",
                "answer_lesioned_control",
                "answer",
                "read_episode_in_answer_context",
            ),
            False,
            "answer_lesioned_control answer differs",
        ),
        (
            (
                "arms",
                "answer_lesioned_control",
                "answer",
                "read_episode_rejoined",
            ),
            False,
            "answer_lesioned_control answer differs",
        ),
        (
            ("gates", "calibration_unchanged"),
            False,
            "gates differ",
        ),
        (
            ("gates", "source_exposed_source_rejoined"),
            False,
            "gates differ",
        ),
        (
            ("gates", "answer_lesioned_control_source_rejoined"),
            False,
            "gates differ",
        ),
        (
            ("gates", "wall_deadline_respected"),
            False,
            "gates differ",
        ),
    ],
)
def test_candidate_only_verification_tampering_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate, _ = _patch_binding(monkeypatch, artifacts)
    verification = _write_candidate_only_verification(
        artifacts,
        candidate,
        mutation=(field_path, value),
    )
    output = tmp_path / "candidate-only-activation"

    with pytest.raises(SkillAdapterActivationError, match=message):
        _prepare_candidate_only(artifacts, output, verification)

    assert not output.exists()


def test_candidate_only_verification_requires_the_exact_added_gate_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate, _ = _patch_binding(monkeypatch, artifacts)
    gates = {
        gate: True
        for gate in activation._CANDIDATE_ONLY_GATE_IDS
        if gate != "wall_deadline_respected"
    }
    verification = _write_candidate_only_verification(
        artifacts,
        candidate,
        mutation=(("gates",), gates),
    )
    output = tmp_path / "candidate-only-activation"

    with pytest.raises(SkillAdapterActivationError, match="gates differ"):
        _prepare_candidate_only(artifacts, output, verification)

    assert not output.exists()


def test_staging_writes_only_an_inert_candidate_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate, _ = _patch_binding(
        monkeypatch,
        artifacts,
        wrong_launch_rank=True,
    )
    output = tmp_path / "staging"
    prepared = _prepare(
        artifacts,
        output,
        staging_only=True,
        served_qualification_path=None,
        expected_served_qualification_sha256=None,
        expected_served_qualification_ref=None,
        rollback_binding_path=None,
        expected_rollback_binding_ref=None,
        active_service_override_path=None,
        expected_active_service_override_sha256=None,
        candidate_container_name=None,
        rollback_container_name=None,
    )

    assert prepared.status == "INERT_STAGING_BINDING"
    assert prepared.binding_ref == candidate.binding_ref
    assert prepared.candidate_override_path is None
    assert prepared.rollback_override_path is None
    assert prepared.plan_path is None
    assert prepared.rollback_binding_ref is None
    assert prepared.plan_sha256 is None
    assert prepared.served_qualification_ref is None
    assert tuple(path.name for path in output.iterdir()) == (
        "candidate-binding.json",
    )
    assert (prepared.binding_path.stat().st_mode & 0o222) == 0


def test_final_activation_without_served_qualification_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="requires one exact served-stack qualification",
    ):
        _prepare(
            artifacts,
            output,
            served_qualification_path=None,
            expected_served_qualification_sha256=None,
            expected_served_qualification_ref=None,
        )
    assert not output.exists()


def test_served_qualification_hash_difference_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="served qualification path is not content-addressed",
    ):
        _prepare(
            artifacts,
            output,
            expected_served_qualification_sha256="0" * 64,
            expected_served_qualification_ref="sha256:" + "0" * 64,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("schema",), "jenny2.wrong.v2", "PASS contract differs"),
        (("passed",), False, "PASS contract differs"),
        (("disposition",), "REJECT", "PASS contract differs"),
        (("source_unchanged",), False, "PASS contract differs"),
        (("library_unchanged",), False, "PASS contract differs"),
        (("bindings_unchanged",), False, "PASS contract differs"),
        (
            ("source_manifest_after_ref",),
            "sha256:" + "4" * 64,
            "source state changed",
        ),
        (
            ("binding_inputs", "weighted_composite", "binding_ref"),
            "sha256:" + "4" * 64,
            "weighted_composite binding differs",
        ),
        (
            (
                "binding_inputs",
                "weighted_composite",
                "selector_qualification_ref",
            ),
            "sha256:" + "4" * 64,
            "weighted_composite binding differs",
        ),
        (
            ("binding_inputs", "parent_control", "binding_ref"),
            "sha256:" + "4" * 64,
            "parent_control binding differs",
        ),
        (
            ("behavioral_gate", "gates", "minimum_micro_accuracy_gain"),
            False,
            "hard gates differ",
        ),
        (
            ("behavioral_gate", "passed"),
            False,
            "behavioral gate differs",
        ),
        (
            (
                "arms",
                "weighted_composite",
                "protocol",
                "structural_reading",
                "library_operation_count",
            ),
            7,
            "reading result differs",
        ),
        (
            (
                "arms",
                "weighted_composite",
                "protocol",
                "structural_reading",
                "all_task_prompts_and_setup_prompts",
            ),
            ["sha256:" + f"{index:064x}" for index in range(1, 14)],
            "reading metrics differ",
        ),
        (
            (
                "arms",
                "weighted_composite",
                "protocol",
                "restart_exact",
            ),
            False,
            "whole-system controls differ",
        ),
    ],
)
def test_served_qualification_semantic_difference_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    _replace_served_qualification_value(artifacts, field_path, value)
    output = tmp_path / "activation"
    with pytest.raises(SkillAdapterActivationError, match=message):
        _prepare(artifacts, output)
    assert not output.exists()


def test_served_qualification_requires_every_exact_hard_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    assert artifacts.served_qualification_file is not None
    payload = json.loads(
        artifacts.served_qualification_file.read_text(encoding="utf-8")
    )
    del payload["behavioral_gate"]["gates"]["minimum_macro_accuracy_gain"]
    artifacts.served_qualification_file.unlink()
    temporary = tmp_path / "missing-gate.tmp"
    _write_canonical_json(temporary, payload)
    digest = _sha(temporary)
    path = tmp_path / f"{digest}.json"
    temporary.replace(path)
    artifacts.served_qualification_file = path
    artifacts.served_qualification_sha = digest
    artifacts.served_qualification_ref = "sha256:" + digest
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="served qualification hard gates differ",
    ):
        _prepare(artifacts, output)
    assert not output.exists()


def test_hash_difference_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(SkillAdapterActivationError, match="training result SHA-256 differs"):
        _prepare(
            artifacts,
            output,
            expected_training_result_sha256="0" * 64,
        )
    assert not output.exists()


def test_wrong_reading_residual_identity_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="reading residual result SHA-256 differs",
    ):
        _prepare(
            artifacts,
            output,
            expected_source_training_result_sha256="0" * 64,
        )
    assert not output.exists()


def test_qualification_hash_difference_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="candidate qualification SHA-256 differs",
    ):
        _prepare(
            artifacts,
            output,
            expected_candidate_qualification_sha256="0" * 64,
        )
    assert not output.exists()


def test_qualification_content_ref_is_recomputed_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _replace_qualification_value(
        artifacts,
        ("final_confirmation", "metrics"),
        {
            "calibration": {
                "selected_weight": artifacts.selected_weight,
                "eligible_weights": [artifacts.selected_weight],
            },
            "final": {"reading_accuracy": 0.98, "retention_accuracy": 0.99},
        },
        recompute_ref=False,
    )
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="candidate qualification ref differs",
    ):
        _prepare(artifacts, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("unexpected",), False, "qualification fields differ"),
        (("schema",), "jenny2.wrong.v1", "qualification contract differs"),
        (("status",), "FAIL", "qualification contract differs"),
        (
            ("protocol_manifest_ref",),
            "sha256:" + "c" * 64,
            "qualification contract differs",
        ),
        (
            ("pre_evaluation_manifest_ref",),
            "sha256:" + "d" * 64,
            "qualification contract differs",
        ),
        (
            ("source_training_result_sha256",),
            "e" * 64,
            "qualification contract differs",
        ),
        (
            ("source_training_metrics_qualifying",),
            True,
            "qualification contract differs",
        ),
        (("runtime_changed",), True, "qualification contract differs"),
        (("live_state_changed",), True, "qualification contract differs"),
        (
            ("selected_candidate", "weight"),
            "1",
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "adapter_model_sha256"),
            "b" * 64,
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "adapter_config_sha256"),
            "b" * 64,
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "composition_manifest_ref"),
            "sha256:" + "b" * 64,
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "composition_manifest_sha256"),
            "b" * 64,
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "training_result_sha256"),
            "b" * 64,
            "qualified candidate selection differs",
        ),
        (
            ("selected_candidate", "rank"),
            8,
            "qualified candidate selection differs",
        ),
        (
            ("final_confirmation", "status"),
            "FAIL",
            "final confirmation differs",
        ),
        (
            ("final_confirmation", "all_hard_gates_passed"),
            False,
            "final confirmation differs",
        ),
        (
            ("final_confirmation", "evaluated_candidate_weights"),
            ["1"],
            "final confirmation differs",
        ),
        (
            ("final_confirmation", "reading_identity_list_sha256"),
            "f" * 64,
            "final confirmation differs",
        ),
        (
            ("final_confirmation", "retention_identity_list_sha256"),
            "f" * 64,
            "final confirmation differs",
        ),
        (
            ("final_confirmation", "gate_results"),
            {
                **_PASSING_GATE_RESULTS,
                "retention_micro_accuracy_noninferiority": False,
            },
            "final hard gates differ",
        ),
        (
            ("final_confirmation", "gate_results"),
            {
                key: value
                for key, value in _PASSING_GATE_RESULTS.items()
                if key != "reading_macro_accuracy_superiority"
            },
            "final hard gates differ",
        ),
        (
            ("final_confirmation", "metrics"),
            {},
            "final metrics differ",
        ),
        (
            ("final_confirmation", "metrics"),
            {
                "calibration": {
                    "selected_weight": "0.25",
                    "eligible_weights": ["0.25", "0.5"],
                },
                "final": {"reading_accuracy": 0.99},
            },
            "metric selection differs",
        ),
    ],
)
def test_qualification_semantic_difference_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    _replace_qualification_value(artifacts, field_path, value)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(SkillAdapterActivationError, match=message):
        _prepare(artifacts, output)
    assert not output.exists()


def test_pinned_protocol_refs_must_match_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="qualification contract differs",
    ):
        _prepare(
            artifacts,
            output,
            expected_calibration_protocol_manifest_ref="sha256:" + "c" * 64,
        )
    assert not output.exists()


def test_qualified_adapter_path_must_be_exact_candidate_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    other_adapter = tmp_path / "other-adapter"
    other_adapter.mkdir()
    _replace_qualification_value(
        artifacts,
        ("selected_candidate", "adapter_path"),
        str(other_adapter),
    )
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="qualified candidate selection differs",
    ):
        _prepare(artifacts, output)
    assert not output.exists()


def test_binding_must_carry_qualification_ref_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts, wrong_qualification_ref=True)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="SGLang candidate binding differs",
    ):
        _prepare(artifacts, output)
    assert not output.exists()


def test_selected_weight_must_match_residual_component_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="component weights differ",
    ):
        _prepare(artifacts, output, expected_selected_weight="0.25")
    assert not output.exists()


def test_selected_weight_must_be_in_frozen_grid_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    output = tmp_path / "activation"
    with pytest.raises(
        SkillAdapterActivationError,
        match="outside the frozen calibration grid",
    ):
        _prepare(artifacts, output, expected_selected_weight="1")
    assert not output.exists()


def test_binding_launch_rank_difference_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts, wrong_launch_rank=True)
    output = tmp_path / "activation"
    with pytest.raises(SkillAdapterActivationError, match="launch LoRA rank differs"):
        _prepare(artifacts, output)
    assert not output.exists()


def test_malformed_override_is_removed_with_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _patch_binding(monkeypatch, artifacts)
    artifacts.override.write_text(
        artifacts.override.read_text()
        + "Environment=JENNY2_SERVED_MODEL=duplicate\n",
        encoding="utf-8",
    )
    output = tmp_path / "activation"
    with pytest.raises(SkillAdapterActivationError, match="exactly once"):
        _prepare(
            artifacts,
            output,
            expected_active_service_override_sha256=_sha(artifacts.override),
        )
    assert not output.exists()


def _cli_common_argv(tmp_path: Path) -> list[str]:
    component = json.dumps(
        {
            "name": "parent",
            "adapter_model_sha256": "1" * 64,
            "adapter_config_sha256": "2" * 64,
            "weight": "1",
        }
    )
    residual = json.dumps(
        {
            "name": "reading",
            "adapter_model_sha256": "3" * 64,
            "adapter_config_sha256": "4" * 64,
            "weight": "0.5",
        }
    )
    return [
        "prepare_jenny2_skill_adapter_activation.py",
        "--composite-run",
        str(tmp_path / "run"),
        "--expected-training-result-sha256",
        "5" * 64,
        "--expected-adapter-model-sha256",
        "6" * 64,
        "--expected-adapter-config-sha256",
        "7" * 64,
        "--expected-manifest-sha256",
        "8" * 64,
        "--expected-manifest-ref",
        "sha256:" + "8" * 64,
        "--expected-source-training-result-sha256",
        "8efcffae18b29b7107ff32848cc4e8a9b29dda982a915bdb5c0dfc814453068d",
        "--expected-component",
        component,
        "--expected-component",
        residual,
        "--expected-rank",
        "16",
        "--expected-profile",
        "reading-skill-v1",
        "--candidate-qualification",
        str(tmp_path / "calibration.json"),
        "--expected-candidate-qualification-sha256",
        "9" * 64,
        "--expected-candidate-qualification-ref",
        "sha256:" + "9" * 64,
        "--expected-calibration-protocol-manifest-ref",
        "sha256:" + "a" * 64,
        "--expected-pre-evaluation-manifest-ref",
        "sha256:" + "b" * 64,
        "--expected-selected-weight",
        "0.5",
        "--output",
        str(tmp_path / "output"),
    ]


def test_cli_staging_mode_passes_no_activation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_prepare(**values: object):
        captured.update(values)
        return SimpleNamespace(
            binding_path=tmp_path / "candidate-binding.json",
            binding_ref="sha256:" + "c" * 64,
            output_root=tmp_path / "output",
            plan_path=None,
            plan_sha256=None,
            rollback_binding_ref=None,
            served_qualification_ref=None,
            status="INERT_STAGING_BINDING",
        )

    monkeypatch.setattr(activation_cli, "prepare_skill_adapter_activation", fake_prepare)
    monkeypatch.setattr(sys, "argv", _cli_common_argv(tmp_path) + ["--staging-only"])
    assert activation_cli.main() == 0
    assert captured["staging_only"] is True
    assert captured["served_qualification_path"] is None
    assert captured["rollback_binding_path"] is None
    output = capsys.readouterr().out
    assert '"status":"INERT_STAGING_BINDING"' in output
    assert '"plan_path":null' in output


def test_cli_final_mode_rejects_missing_served_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _cli_common_argv(tmp_path) + [
        "--rollback-binding",
        str(tmp_path / "rollback.json"),
        "--expected-rollback-binding-ref",
        "sha256:" + "c" * 64,
        "--active-service-override",
        str(tmp_path / "override.conf"),
        "--expected-active-service-override-sha256",
        "d" * 64,
        "--candidate-container-name",
        "candidate",
        "--rollback-container-name",
        "rollback",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        activation_cli.main()
    assert exc.value.code == 2
