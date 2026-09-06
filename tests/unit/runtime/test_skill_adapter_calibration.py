from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file

from angler.runtime.skill_adapter_calibration import (
    CompiledCandidate,
    ComponentIdentity,
    DatasetIdentity,
    EvaluationRow,
    FileIdentity,
    FrozenCalibrationProtocol,
    LORA_TARGET_MODULES,
    PartitionFingerprint,
    READING_SKILLS,
    RETENTION_SKILLS,
    RowSufficientStatistic,
    WEIGHT_GRID,
    compiler_contract_payload,
    evaluator_contract_payload,
    run_skill_adapter_calibration,
    _verify_candidate_tensor_slices,
)
from scripts.run_jenny2_skill_adapter_calibration import (
    _EncodedEvaluationRow,
    _length_bucketed_batches,
    _right_padded_batch_payload,
    _row_statistics_from_logits,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fake_evaluator_identity(
    protocol: FrozenCalibrationProtocol,
) -> dict[str, object]:
    import angler.runtime.skill_adapter_calibration as calibration_module

    entrypoint = Path(__file__).resolve()
    orchestrator = Path(calibration_module.__file__).resolve()
    return {
        "schema": "jenny2.skill-adapter-evaluator-identity.v1",
        "entrypoint": {"path": str(entrypoint), "sha256": _sha(entrypoint)},
        "orchestrator": {
            "path": str(orchestrator),
            "sha256": _sha(orchestrator),
        },
        "python_runtime": sys.version,
        "dependencies": {"cpu-fake-backend": "1"},
        "contract": evaluator_contract_payload(protocol),
        "compiler": {
            "schema": "jenny2.skill-adapter-compiler-identity.v1",
            "implementation_files": [
                {"path": str(entrypoint), "sha256": _sha(entrypoint)}
            ],
            "dependencies": {
                "numpy": "test",
                "safetensors": "test",
                "torch": str(torch.__version__),
            },
            "contract": compiler_contract_payload(protocol),
        },
    }


def _component(root: Path, name: str) -> ComponentIdentity:
    path = root / name
    path.mkdir()
    _write_json(
        path / "adapter_config.json",
        {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "use_rslora": False,
        },
    )
    offset = 1.0 if name == "prior_competence" else 101.0
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            prefix + ".lora_A.weight": (
                torch.arange(24, dtype=torch.float32).reshape(8, 3) + offset
            ),
            prefix + ".lora_B.weight": (
                torch.arange(32, dtype=torch.float32).reshape(4, 8) + offset
            ),
        },
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    return ComponentIdentity(
        name=name,
        path=path,
        adapter_model_sha256=_sha(path / "adapter_model.safetensors"),
        adapter_config_sha256=_sha(path / "adapter_config.json"),
    )


def _row(identity: str, skill: str, boundary: str | None = "public_cortex") -> dict:
    value = {
        "identity": identity,
        "skill": skill,
        "messages": [
            {"role": "system", "content": "bounded synthetic fixture"},
            {"role": "user", "content": identity},
            {"role": "assistant", "content": "expected"},
        ],
        "expected": "expected",
        "schema": "fixture.v1",
        "split": "eval",
    }
    if boundary is not None:
        value["boundary"] = boundary
    return value


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _partition_fingerprint(rows: list[dict], raw_lines: list[bytes]) -> PartitionFingerprint:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["skill"]] = counts.get(row["skill"], 0) + 1
    return PartitionFingerprint(
        count=len(rows),
        raw_jsonl_sha256=hashlib.sha256(b"".join(raw_lines)).hexdigest(),
        identity_list_sha256=hashlib.sha256(
            "".join(row["identity"] + "\n" for row in rows).encode()
        ).hexdigest(),
        skill_counts=counts,
    )


@dataclass
class _Fixture:
    protocol: FrozenCalibrationProtocol
    output: Path
    evaluator_created: bool = False
    compiler_calls: list[str] | None = None
    evaluation_calls: list[tuple[str, str | None, str, str]] | None = None

    def __post_init__(self) -> None:
        self.compiler_calls = []
        self.evaluation_calls = []


def _fixture(tmp_path: Path) -> _Fixture:
    parent = _component(tmp_path, "prior_competence")
    residual = _component(tmp_path, "library_reading")
    reading_path = tmp_path / "reading.jsonl"
    reading_consumed = [
        _row(f"reading-retired-{i}", skill)
        for i, skill in enumerate(READING_SKILLS)
    ]
    reading_calibration = [
        _row(f"reading-cal-{cycle}-{skill}", skill)
        for cycle in range(4)
        for skill in READING_SKILLS
    ]
    reading_final = [
        _row(f"reading-final-{cycle}-{skill}", skill)
        for cycle in range(2)
        for skill in READING_SKILLS
    ]
    reading_rows = reading_consumed + reading_calibration + reading_final
    _write_rows(reading_path, reading_rows)
    reading_raw = reading_path.read_bytes().splitlines(keepends=True)

    retention_path = tmp_path / "retention.jsonl"
    retention_consumed = [
        _row(f"retention-retired-{i}", skill, None)
        for i, skill in enumerate(RETENTION_SKILLS)
    ]
    retention_calibration = [
        _row(f"retention-cal-{cycle}-{skill}", skill, None)
        for cycle in range(4)
        for skill in RETENTION_SKILLS
    ]
    retention_final = [
        _row(f"retention-final-{cycle}-{skill}", skill, None)
        for cycle in range(2)
        for skill in RETENTION_SKILLS
    ]
    retention_rows = retention_consumed + retention_calibration + retention_final
    _write_rows(retention_path, retention_rows)
    retention_raw = retention_path.read_bytes().splitlines(keepends=True)

    base = tmp_path / "base"
    base.mkdir()
    (base / "shard-a.safetensors").write_bytes(b"a")
    (base / "shard-b.safetensors").write_bytes(b"b")
    _write_json(
        base / "model.safetensors.index.json",
        {"weight_map": {"a": "shard-a.safetensors", "b": "shard-b.safetensors"}},
    )
    _write_json(base / "config.json", {"model": "fixture"})
    _write_json(base / "tokenizer.json", {"tokenizer": "fixture"})
    _write_json(base / "tokenizer_config.json", {"chat_template": "fixture"})
    (base / "chat_template.jinja").write_text("fixture", encoding="utf-8")
    base_files = tuple(
        FileIdentity(Path(name), _sha(base / name))
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "shard-a.safetensors",
            "shard-b.safetensors",
        )
    )

    result_path = tmp_path / "source-training-result.json"
    result = {
        "schema": "jenny2.qlora-training.v1",
        "status": "PASS",
        "base_revision": "e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c",
        "base_config_sha256": _sha(base / "config.json"),
        "base_index_sha256": _sha(base / "model.safetensors.index.json"),
        "training_profile": "reading-skill-v1",
        "training_sha256": "1" * 64,
        "curriculum_manifest_sha256": "2" * 64,
        "text_only_training": True,
        "lora_scaling": "standard-alpha-over-r",
        "adapter_training_mode": "residual-skill",
        "parent_adapter_name": parent.name,
        "residual_skill_name": residual.name,
        "adapter_model_sha256": residual.adapter_model_sha256,
        "adapter_config_sha256": residual.adapter_config_sha256,
        "adapter_path": str(residual.path),
        "standalone_skill_adapter_path": str(residual.path),
        "initial_adapter_path": str(parent.path),
        "initial_adapter_model_sha256": parent.adapter_model_sha256,
        "initial_adapter_config_sha256": parent.adapter_config_sha256,
        "evaluation_sha256": _sha(reading_path),
        "evaluation_limit": len(reading_consumed),
        "retention_evaluation_sha256": _sha(retention_path),
        "retention_evaluation_limit": len(retention_consumed),
        "maximum_tokens": 3072,
        "rank": 8,
        "target_modules": list(LORA_TARGET_MODULES),
        "parent_parameter_fingerprint_unchanged": True,
        "composition_operator": "peft-additive-active-adapter-sum",
        "active_adapter_components": [parent.name, residual.name],
        "held_out_teacher_forced_evaluation": {"retired": True},
        "retention_teacher_forced_evaluation": {"retired": True},
        "max_steps": 4,
        "steps_completed": 4,
    }
    _write_json(result_path, result)
    protocol = FrozenCalibrationProtocol(
        source_training_result=FileIdentity(result_path, _sha(result_path)),
        parent=parent,
        residual=residual,
        base_root=base,
        base_files=base_files,
        reading=DatasetIdentity(
            source=FileIdentity(reading_path, _sha(reading_path)),
            consumed_count=len(reading_consumed),
            calibration=_partition_fingerprint(
                reading_calibration,
                reading_raw[len(reading_consumed) : len(reading_consumed) + 60],
            ),
            final_test=_partition_fingerprint(reading_final, reading_raw[-30:]),
        ),
        retention=DatasetIdentity(
            source=FileIdentity(retention_path, _sha(retention_path)),
            consumed_count=len(retention_consumed),
            calibration=_partition_fingerprint(
                retention_calibration,
                retention_raw[len(retention_consumed) : len(retention_consumed) + 72],
            ),
            final_test=_partition_fingerprint(retention_final, retention_raw[-36:]),
        ),
    )
    return _Fixture(protocol=protocol, output=tmp_path / "evidence")


def _compile(fixture: _Fixture, weight: str, output: Path) -> CompiledCandidate:
    assert fixture.evaluator_created is False
    assert fixture.compiler_calls is not None
    fixture.compiler_calls.append(weight)
    output.mkdir()
    adapter = output / "adapter"
    adapter.mkdir()
    model_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    parent_tensors = load_file(
        str(fixture.protocol.parent.path / "adapter_model.safetensors"),
        device="cpu",
    )
    residual_tensors = load_file(
        str(fixture.protocol.residual.path / "adapter_model.safetensors"),
        device="cpu",
    )
    applied_weight = float(torch.tensor(float(weight), dtype=torch.float32).item())
    composite_tensors: dict[str, torch.Tensor] = {}
    for key in sorted(parent_tensors):
        if key.endswith(".lora_A.weight"):
            composite_tensors[key] = torch.cat(
                (parent_tensors[key], residual_tensors[key] * applied_weight), dim=0
            )
        else:
            composite_tensors[key] = torch.cat(
                (parent_tensors[key], residual_tensors[key]), dim=1
            )
    save_file(composite_tensors, str(model_path), metadata={"format": "pt"})
    _write_json(
        config_path,
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": str(fixture.protocol.base_root),
            "r": 16,
            "lora_alpha": 32,
            "inference_mode": True,
            "lora_dropout": 0.0,
            "use_rslora": False,
            "bias": "none",
            "target_modules": sorted(LORA_TARGET_MODULES),
        },
    )
    model_sha = _sha(model_path)
    config_sha = _sha(config_path)
    unsigned_manifest = {
        "schema": "jenny2.skill-adapter-composition.v1",
        "status": "PASS",
        "algorithm": "exact-lora-rank-concatenation-v1",
        "foundation_weights_changed": False,
        "query_conditioned_adapter_selection": False,
        "source_training_result_sha256": fixture.protocol.source_training_result.sha256,
        "components": [
            {
                "ordinal": 0,
                "name": fixture.protocol.parent.name,
                "adapter_model_sha256": fixture.protocol.parent.adapter_model_sha256,
                "adapter_config_sha256": fixture.protocol.parent.adapter_config_sha256,
                "requested_weight": "1",
            },
            {
                "ordinal": 1,
                "name": fixture.protocol.residual.name,
                "adapter_model_sha256": fixture.protocol.residual.adapter_model_sha256,
                "adapter_config_sha256": fixture.protocol.residual.adapter_config_sha256,
                "requested_weight": weight,
            },
        ],
        "composite": {
            "rank": 16,
            "alpha": 32,
            "alpha_over_rank": 2,
            "adapter_model_sha256": model_sha,
            "adapter_config_sha256": config_sha,
        },
        "shape_audit": [{"exact_rank_slice_audit": True}],
    }
    manifest_ref = "sha256:" + hashlib.sha256(
        _canonical(unsigned_manifest)
    ).hexdigest()
    manifest_path = output / "composition-manifest.json"
    _write_json(manifest_path, {**unsigned_manifest, "manifest_ref": manifest_ref})
    manifest_sha = _sha(manifest_path)
    result_path = output / "training-result.json"
    source_result = json.loads(
        fixture.protocol.source_training_result.path.read_text(encoding="utf-8")
    )
    _write_json(
        result_path,
        {
            **source_result,
            "schema": "jenny2.qlora-training.v1",
            "status": "PASS",
            "adapter_artifact_kind": "exact-rank-concatenated-skill-composite",
            "adapter_model_sha256": model_sha,
            "adapter_config_sha256": config_sha,
            "composition_manifest_ref": manifest_ref,
            "composition_manifest_sha256": manifest_sha,
            "composition_source_training_result_sha256": (
                fixture.protocol.source_training_result.sha256
            ),
            "rank": 16,
            # These deliberately copied source metrics cannot qualify the candidate.
            "held_out_teacher_forced_evaluation": {"retired_weight_1": True},
        },
    )
    return CompiledCandidate(
        weight=weight,
        output_root=output,
        adapter_path=adapter,
        manifest_path=manifest_path,
        training_result_path=result_path,
        adapter_model_sha256=model_sha,
        adapter_config_sha256=config_sha,
        manifest_ref=manifest_ref,
        manifest_sha256=manifest_sha,
        training_result_sha256=_sha(result_path),
        rank=16,
    )


class _Backend:
    def __init__(self, fixture: _Fixture, mode: str) -> None:
        self.fixture = fixture
        self.mode = mode

    def evaluate(
        self, target, rows, *, phase: str, suite: str, maximum_tokens: int
    ):
        assert maximum_tokens == 3072
        assert self.fixture.evaluation_calls is not None
        if phase == "final":
            selection_path = self.fixture.output / "calibration-selection.json"
            assert selection_path.is_file()
            assert os.stat(selection_path).st_mode & 0o222 == 0
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            assert selection["selected_weight"] == "0.25"
            assert selection["authorized_final_candidate_weights"] == ["0.25"]
            assert selection["final_test"][
                "metric_evaluation_started_before_selection_seal"
            ] is False
        self.fixture.evaluation_calls.append((target.name, target.weight, phase, suite))
        if phase == "final" and target.name == "weighted_composite":
            assert target.weight == "0.25", "no fallback candidate may see final rows"
        accuracy, nll = self._values(target, phase, suite)
        tokens = 1000
        correct = int(accuracy * tokens)
        observed = tuple(
            RowSufficientStatistic(
                identity=row.identity,
                skill=row.skill,
                boundary=row.boundary,
                supervised_tokens=tokens,
                correct_tokens=correct,
                nll_sum=nll * tokens,
            )
            for row in rows
        )
        if (
            self.mode == "backend_mutates"
            and phase == "final"
            and target.name == "weighted_composite"
            and suite == "retention"
        ):
            model = target.adapter_path / "adapter_model.safetensors"
            os.chmod(model, 0o600)
            model.write_bytes(b"changed-during-final-evaluation")
        return observed

    def close(self) -> None:
        pass

    def _values(self, target, phase: str, suite: str) -> tuple[float, float]:
        if target.name == "parent_control":
            if self.mode == "parent_invalid" and suite == "retention":
                return 0.98, 0.01
            return (0.60, 1.0) if suite == "reading" else (1.0, 0.01)
        assert target.weight is not None
        if suite == "reading":
            accuracies = {
                "0.0625": 0.65,
                "0.125": 0.72,
                "0.25": 0.80 if phase == "calibration" else 0.78,
                "0.375": 0.85,
                "0.5": 0.88,
                "0.75": 0.90,
            }
            return accuracies[target.weight], 0.50
        if self.mode == "no_safe":
            return 0.98, 0.02
        if phase == "final" and self.mode == "final_fail":
            return 0.98, 0.02
        retention = {
            "0.0625": 1.0,
            "0.125": 0.995,
            "0.25": 0.99,
            "0.375": 0.98,
            "0.5": 0.97,
            "0.75": 0.95,
        }
        return retention[target.weight], 0.02


def _run(fixture: _Fixture, mode: str = "pass"):
    def compiler(weight: str, output: Path):
        return _compile(fixture, weight, output)

    def factory():
        # This assertion proves candidate hashes and row partitions were persisted
        # before construction of the evaluator that could expose any metric.
        pre = fixture.output / "pre-evaluation-manifest.json"
        assert pre.is_file()
        assert os.stat(pre).st_mode & 0o222 == 0
        payload = json.loads(pre.read_text(encoding="utf-8"))
        assert payload["status"] == "FROZEN_BEFORE_METRICS"
        assert payload["candidate_count"] == 6
        assert [item["weight"] for item in payload["candidates"]] == list(
            WEIGHT_GRID
        )
        assert payload["metrics_observed_before_seal"] is False
        assert payload["evaluator_identity"]["contract"][
            "served_sglang_nvfp4_confirmation_required"
        ] is True
        fixture.evaluator_created = True
        return _Backend(fixture, mode)

    return run_skill_adapter_calibration(
        output_root=fixture.output,
        compiler=compiler,
        evaluator_factory=factory,
        evaluator_identity=_fake_evaluator_identity(fixture.protocol),
        protocol=fixture.protocol,
    )


def test_pass_seals_all_candidates_selects_once_and_matches_activation_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(tmp_path)
    run = _run(fixture)

    assert run.status == "PASS"
    assert fixture.compiler_calls == list(WEIGHT_GRID)
    assert fixture.evaluation_calls is not None
    final_candidate_calls = [
        call
        for call in fixture.evaluation_calls
        if call[2] == "final" and call[0] == "weighted_composite"
    ]
    assert final_candidate_calls == [
        ("weighted_composite", "0.25", "final", "reading"),
        ("weighted_composite", "0.25", "final", "retention"),
    ]
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert run.selection_manifest_path is not None
    assert run.selection_manifest_ref is not None
    assert os.stat(run.selection_manifest_path).st_mode & 0o222 == 0
    selection = json.loads(
        run.selection_manifest_path.read_text(encoding="utf-8")
    )
    assert selection["status"] == "SELECTED"
    assert selection["selected_weight"] == "0.25"
    assert selection["authorized_final_candidate_weights"] == ["0.25"]
    assert selection["final_test"][
        "metric_evaluation_started_before_selection_seal"
    ] is False
    assert selection["full_calibration_statistics"]["selected_weight"] == "0.25"
    assert set(result) == {
        "schema",
        "status",
        "qualification_ref",
        "protocol_manifest_ref",
        "pre_evaluation_manifest_ref",
        "source_training_result_sha256",
        "source_training_metrics_qualifying",
        "selected_candidate",
        "final_confirmation",
        "runtime_changed",
        "live_state_changed",
    }
    assert result["source_training_metrics_qualifying"] is False
    assert result["selected_candidate"]["weight"] == "0.25"
    confirmation = result["final_confirmation"]
    assert confirmation["evaluated_candidate_weights"] == ["0.25"]
    assert confirmation["all_hard_gates_passed"] is True
    assert confirmation["metrics"]["calibration"]["selected_weight"] == "0.25"
    assert (
        confirmation["metrics"]["calibration"]["calibration_selection_ref"]
        == run.selection_manifest_ref
    )
    assert set(confirmation["metrics"]) == {"calibration", "final"}
    assert all(confirmation["gate_results"].values())
    assert confirmation["metrics"]["final"]["weighted_composite"]["reading"][
        "row_sufficient_statistics"
    ]
    unsigned = dict(result)
    observed_ref = unsigned.pop("qualification_ref")
    assert observed_ref == "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()

    # Exercise the consuming activation contract directly so the producer and
    # validator cannot silently drift while both unit suites still pass alone.
    from angler.runtime import skill_adapter_activation

    monkeypatch.setattr(
        skill_adapter_activation,
        "_READING_IDENTITY_LIST_SHA256",
        fixture.protocol.reading.final_test.identity_list_sha256,
    )
    monkeypatch.setattr(
        skill_adapter_activation,
        "_RETENTION_IDENTITY_LIST_SHA256",
        fixture.protocol.retention.final_test.identity_list_sha256,
    )

    selected = result["selected_candidate"]
    skill_adapter_activation._validate_candidate_qualification(
        result,
        adapter_path=Path(selected["adapter_path"]),
        expected_ref=result["qualification_ref"],
        expected_protocol_manifest_ref=result["protocol_manifest_ref"],
        expected_pre_evaluation_manifest_ref=result["pre_evaluation_manifest_ref"],
        expected_source_training_result_sha256=(
            fixture.protocol.source_training_result.sha256
        ),
        expected_selected_weight="0.25",
        expected_model_sha256=selected["adapter_model_sha256"],
        expected_config_sha256=selected["adapter_config_sha256"],
        expected_manifest_ref=selected["composition_manifest_ref"],
        expected_manifest_sha256=selected["composition_manifest_sha256"],
        expected_training_result_sha256=selected["training_result_sha256"],
        expected_rank=16,
    )

    # Exercise the real binding package validator too; a calibration PASS must
    # not name an artifact that the served-stack binding cannot consume.
    from angler.runtime import jenny2_runtime

    monkeypatch.setattr(
        jenny2_runtime, "QWEN38_SOURCE_MODEL_PATH", fixture.protocol.base_root
    )
    monkeypatch.setattr(
        jenny2_runtime,
        "QWEN38_SOURCE_REVISION",
        "e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c",
    )
    monkeypatch.setattr(
        jenny2_runtime,
        "QWEN38_SOURCE_CONFIG_SHA256",
        next(
            item.sha256
            for item in fixture.protocol.base_files
            if item.path.name == "config.json"
        ),
    )
    monkeypatch.setattr(
        jenny2_runtime,
        "QWEN38_SOURCE_INDEX_SHA256",
        next(
            item.sha256
            for item in fixture.protocol.base_files
            if item.path.name == "model.safetensors.index.json"
        ),
    )
    monkeypatch.setattr(
        jenny2_runtime, "QWEN38_LORA_TARGET_MODULES", LORA_TARGET_MODULES
    )
    candidate_run = Path(selected["adapter_path"]).parent
    packaged_result = json.loads(
        (candidate_run / "training-result.json").read_text(encoding="utf-8")
    )
    packaged_config = json.loads(
        (candidate_run / "adapter" / "adapter_config.json").read_text(
            encoding="utf-8"
        )
    )
    jenny2_runtime.SGLangLoRABinding._validate_training_result(
        packaged_result,
        adapter_config=packaged_config,
        adapter_model_sha256=selected["adapter_model_sha256"],
        adapter_config_sha256=selected["adapter_config_sha256"],
    )


def test_no_safe_weight_preserves_final_without_evaluating_it(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run = _run(fixture, mode="no_safe")

    assert run.status == "NO_SAFE_WEIGHT"
    assert fixture.evaluation_calls is not None
    assert all(call[2] == "calibration" for call in fixture.evaluation_calls)
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert result["selected_candidate"] is None
    assert result["final_confirmation"]["status"] == "NOT_RUN"
    assert result["final_confirmation"]["evaluated_candidate_weights"] == []
    assert run.selection_manifest_path is not None
    selection = json.loads(
        run.selection_manifest_path.read_text(encoding="utf-8")
    )
    assert selection["status"] == "NO_SAFE_WEIGHT"
    assert selection["selected_weight"] is None
    assert selection["authorized_final_candidate_weights"] == []


def test_final_failure_does_not_fallback_to_another_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run = _run(fixture, mode="final_fail")

    assert run.status == "WEIGHTED_COMPOSITION_REJECTED"
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert result["selected_candidate"]["weight"] == "0.25"
    assert result["final_confirmation"]["evaluated_candidate_weights"] == ["0.25"]
    assert result["final_confirmation"]["all_hard_gates_passed"] is False
    assert fixture.evaluation_calls is not None
    assert {
        call[1]
        for call in fixture.evaluation_calls
        if call[2] == "final" and call[0] == "weighted_composite"
    } == {"0.25"}


def test_source_hash_difference_fails_before_compilation_or_evaluator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    (fixture.protocol.residual.path / "adapter_model.safetensors").write_bytes(
        b"changed"
    )
    run = _run(fixture)

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.compiler_calls == []
    assert fixture.evaluator_created is False
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert result["runtime_changed"] is False
    assert result["live_state_changed"] is False
    assert "SHA-256 differs" in result["error"]["message"]


def test_invalid_parent_control_is_inconclusive_and_selects_nothing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    run = _run(fixture, mode="parent_invalid")

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.evaluation_calls == [
        ("parent_control", None, "calibration", "reading"),
        ("parent_control", None, "calibration", "retention"),
    ]
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert result["selected_candidate"] is None
    assert "parent control failed" in result["error"]["message"]


def test_base_index_requires_every_and_only_bound_shards(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reduced = tuple(
        item
        for item in fixture.protocol.base_files
        if item.path.name != "shard-b.safetensors"
    )
    fixture.protocol = FrozenCalibrationProtocol(
        source_training_result=fixture.protocol.source_training_result,
        parent=fixture.protocol.parent,
        residual=fixture.protocol.residual,
        base_root=fixture.protocol.base_root,
        base_files=reduced,
        reading=fixture.protocol.reading,
        retention=fixture.protocol.retention,
    )
    run = _run(fixture)

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.compiler_calls == []
    assert fixture.evaluator_created is False
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert "bound base shard set differs" in result["error"]["message"]


def test_compiler_cannot_mutate_frozen_protocol_before_preseal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def compiler(weight: str, output: Path):
        candidate = _compile(fixture, weight, output)
        protocol_path = fixture.output / "protocol-manifest.json"
        os.chmod(protocol_path, 0o600)
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        payload["status"] = "MUTATED"
        _write_json(protocol_path, payload)
        return candidate

    run = run_skill_adapter_calibration(
        output_root=fixture.output,
        compiler=compiler,
        evaluator_factory=lambda: (_ for _ in ()).throw(
            AssertionError("evaluator must not be constructed")
        ),
        evaluator_identity=_fake_evaluator_identity(fixture.protocol),
        protocol=fixture.protocol,
    )

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.compiler_calls == ["0.0625"]
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert "protocol manifest seal differs" in result["error"]["message"]


def test_compiler_base_mutation_is_detected_before_evaluator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def compiler(weight: str, output: Path):
        candidate = _compile(fixture, weight, output)
        if weight == WEIGHT_GRID[-1]:
            (fixture.protocol.base_root / "shard-a.safetensors").write_bytes(
                b"mutated"
            )
        return candidate

    run = run_skill_adapter_calibration(
        output_root=fixture.output,
        compiler=compiler,
        evaluator_factory=lambda: (_ for _ in ()).throw(
            AssertionError("evaluator must not be constructed")
        ),
        evaluator_identity=_fake_evaluator_identity(fixture.protocol),
        protocol=fixture.protocol,
    )

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.compiler_calls == list(WEIGHT_GRID)
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert "SHA-256 differs" in result["error"]["message"]


def test_compiler_must_return_the_requested_unique_grid_arm(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def compiler(weight: str, output: Path):
        candidate = _compile(fixture, weight, output)
        if weight == "0.125":
            return CompiledCandidate(
                weight="0.0625",
                output_root=candidate.output_root,
                adapter_path=candidate.adapter_path,
                manifest_path=candidate.manifest_path,
                training_result_path=candidate.training_result_path,
                adapter_model_sha256=candidate.adapter_model_sha256,
                adapter_config_sha256=candidate.adapter_config_sha256,
                manifest_ref=candidate.manifest_ref,
                manifest_sha256=candidate.manifest_sha256,
                training_result_sha256=candidate.training_result_sha256,
                rank=candidate.rank,
            )
        return candidate

    run = run_skill_adapter_calibration(
        output_root=fixture.output,
        compiler=compiler,
        evaluator_factory=lambda: (_ for _ in ()).throw(
            AssertionError("evaluator must not be constructed")
        ),
        evaluator_identity=_fake_evaluator_identity(fixture.protocol),
        protocol=fixture.protocol,
    )

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    assert fixture.compiler_calls == ["0.0625", "0.125"]
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert "compiled candidate weight differs" in result["error"]["message"]


def test_backend_mutation_cannot_leave_a_pass_result(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run = _run(fixture, mode="backend_mutates")

    assert run.status == "INCONCLUSIVE_FAIL_CLOSED"
    result = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert result["final_confirmation"]["all_hard_gates_passed"] is False
    assert "SHA-256 differs" in result["error"]["message"]


def test_length_bucketed_microbatch_preserves_padding_labels_and_row_statistics(
) -> None:
    rows = (
        EvaluationRow(
            "row-0", "reading", "public_cortex", 1, b"{}\n", {}
        ),
        EvaluationRow(
            "row-1", "retention", None, 2, b"{}\n", {}
        ),
        EvaluationRow(
            "row-2", "reading", "public_cortex", 3, b"{}\n", {}
        ),
    )
    encoded = (
        _EncodedEvaluationRow(0, rows[0], (10, 11, 12, 13, 14), (-100, -100, 1, 2, 3)),
        _EncodedEvaluationRow(1, rows[1], (20, 21, 22), (-100, 2, 1)),
        _EncodedEvaluationRow(2, rows[2], (30, 31, 32, 33), (-100, -100, -100, 3)),
    )
    buckets = _length_bucketed_batches(encoded, batch_size=2)
    assert [[row.original_index for row in batch] for batch in buckets] == [
        [1, 2],
        [0],
    ]

    payload = _right_padded_batch_payload(buckets[0], pad_token_id=0)
    assert payload["input_ids"] == [[20, 21, 22, 0], [30, 31, 32, 33]]
    assert payload["attention_mask"] == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert payload["labels"] == [
        [-100, 2, 1, -100],
        [-100, -100, -100, 3],
    ]
    assert payload["shift_labels"] == [
        [2, 1, -100, -100],
        [-100, -100, 3, -100],
    ]
    assert payload["supervised_tokens"] == [2, 1]
    assert payload["logits_to_keep"] == 4

    targets = torch.tensor(payload["shift_labels"], dtype=torch.long)
    logits = torch.zeros((2, 4, 5), dtype=torch.float32)
    logits[0, 0, 2] = 4.0  # correct target 2
    logits[0, 1, 0] = 4.0  # incorrect target 1
    logits[1, 2, 3] = 4.0  # correct target 3
    observed = dict(
        _row_statistics_from_logits(
            rows=buckets[0],
            logits=logits,
            shift_labels=targets,
            functional=functional,
            torch_module=torch,
        )
    )
    assert set(observed) == {1, 2}
    assert observed[1].identity == "row-1"
    assert observed[1].supervised_tokens == 2
    assert observed[1].correct_tokens == 1
    assert observed[2].identity == "row-2"
    assert observed[2].supervised_tokens == 1
    assert observed[2].correct_tokens == 1
    token_losses = functional.cross_entropy(
        logits.reshape(-1, 5),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)
    assert observed[1].nll_sum == pytest.approx(
        float(token_losses[0, :2].sum(dtype=torch.float64))
    )
    assert observed[2].nll_sum == pytest.approx(
        float(token_losses[1, 2].to(dtype=torch.float64))
    )


def test_independent_tensor_verifier_rejects_wrong_weighted_rank_slice(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = _compile(fixture, "0.25", tmp_path / "candidate-proof")
    model_path = candidate.adapter_path / "adapter_model.safetensors"
    tensors = load_file(str(model_path), device="cpu")
    a_key = next(key for key in tensors if key.endswith(".lora_A.weight"))
    tensors[a_key][fixture.protocol.parent.rank :] += 1.0
    save_file(tensors, str(model_path), metadata={"format": "pt"})

    with pytest.raises(
        ValueError,
        match="tensor slices do not equal declared weighted sources",
    ):
        _verify_candidate_tensor_slices(candidate, fixture.protocol)
