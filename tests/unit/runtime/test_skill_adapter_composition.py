from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from angler.runtime.skill_adapter_composition import (
    SkillAdapterComponent,
    SkillAdapterCompositionError,
    compose_skill_adapters,
)


TARGETS = ["q_proj", "up_proj"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter(
    root: Path,
    name: str,
    *,
    rank: int = 2,
    offset: float = 0.0,
    base: str = "/models/frozen-qwen",
    targets: list[str] | None = None,
    nonfinite: bool = False,
) -> SkillAdapterComponent:
    path = root / name
    path.mkdir()
    config = {
        "alpha_pattern": {},
        "base_model_name_or_path": base,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "loftq_config": {},
        "lora_alpha": rank * 2,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.20.0",
        "r": rank,
        "rank_pattern": {},
        "revision": "frozen-revision",
        "target_modules": TARGETS if targets is None else targets,
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    (path / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix_q = "base_model.model.model.layers.0.self_attn.q_proj"
    prefix_u = "base_model.model.model.layers.0.mlp.up_proj"
    tensors = {
        prefix_q + ".lora_A.weight": torch.arange(
            rank * 3, dtype=torch.float32
        ).reshape(rank, 3)
        / 17
        + offset,
        prefix_q + ".lora_B.weight": torch.arange(
            4 * rank, dtype=torch.float32
        ).reshape(4, rank)
        / 13
        + offset,
        prefix_u + ".lora_A.weight": torch.arange(
            rank * 4, dtype=torch.float32
        ).reshape(rank, 4)
        / 19
        + offset,
        prefix_u + ".lora_B.weight": torch.arange(
            5 * rank, dtype=torch.float32
        ).reshape(5, rank)
        / 23
        + offset,
    }
    if nonfinite:
        tensors[prefix_q + ".lora_A.weight"][0, 0] = float("nan")
    save_file(tensors, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    return SkillAdapterComponent(
        name=name,
        path=path,
        adapter_model_sha256=_sha(path / "adapter_model.safetensors"),
        adapter_config_sha256=_sha(path / "adapter_config.json"),
        weight="1",
    )


def _training_result(root: Path, trained: SkillAdapterComponent, *, rank: int = 2) -> Path:
    payload = {
        "schema": "jenny2.qlora-training.v1",
        "status": "PASS",
        "base_revision": "frozen-revision",
        "base_config_sha256": "1" * 64,
        "base_index_sha256": "2" * 64,
        "text_only_training": True,
        "lora_scaling": "standard-alpha-over-r",
        "training_sha256": "3" * 64,
        "evaluation_sha256": "4" * 64,
        "curriculum_manifest_sha256": "5" * 64,
        "training_profile": "reading-skill-v1",
        "adapter_model_sha256": trained.adapter_model_sha256,
        "adapter_config_sha256": trained.adapter_config_sha256,
        "target_modules": TARGETS,
        "rank": rank,
        "max_steps": 4,
        "steps_completed": 4,
        "held_out_teacher_forced_evaluation": {"examples": 2},
        "latest_checkpoint": str(Path(trained.path)),
        "latest_checkpoint_sha256": trained.adapter_model_sha256,
    }
    path = root / "source-training-result.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _with_weight(component: SkillAdapterComponent, weight: str) -> SkillAdapterComponent:
    return SkillAdapterComponent(
        name=component.name,
        path=component.path,
        adapter_model_sha256=component.adapter_model_sha256,
        adapter_config_sha256=component.adapter_config_sha256,
        weight=weight,
    )


def test_exact_rank_concatenation_and_existing_binding_shape(tmp_path: Path) -> None:
    parent = _adapter(tmp_path, "initial", offset=0.1)
    reading = _adapter(tmp_path, "reading", offset=0.3)
    result = _training_result(tmp_path, reading)
    composition = compose_skill_adapters(
        [parent, _with_weight(reading, "0.5")],
        training_result_source=result,
        expected_training_result_sha256=_sha(result),
        output_root=tmp_path / "compiled",
    )

    config = json.loads((composition.adapter_path / "adapter_config.json").read_text())
    assert config["r"] == 4
    assert config["lora_alpha"] == 8
    assert config["target_modules"] == sorted(TARGETS)
    compiled = load_file(str(composition.adapter_path / "adapter_model.safetensors"))
    first = load_file(str(Path(parent.path) / "adapter_model.safetensors"))
    second = load_file(str(Path(reading.path) / "adapter_model.safetensors"))
    for a_key in sorted(key for key in first if key.endswith(".lora_A.weight")):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        assert torch.equal(compiled[a_key][:2], first[a_key])
        assert torch.equal(compiled[a_key][2:], second[a_key] * 0.5)
        assert torch.equal(compiled[b_key][:, :2], first[b_key])
        assert torch.equal(compiled[b_key][:, 2:], second[b_key])
        vector = torch.linspace(-0.7, 0.9, first[a_key].shape[1])
        expected = 2 * (first[b_key] @ (first[a_key] @ vector))
        expected += 0.5 * 2 * (second[b_key] @ (second[a_key] @ vector))
        observed = 2 * (compiled[b_key] @ (compiled[a_key] @ vector))
        assert torch.allclose(observed, expected, rtol=1e-5, atol=1e-6)

    manifest = json.loads(composition.manifest_path.read_text())
    assert manifest["status"] == "PASS"
    assert manifest["trained_component"] == "reading"
    assert manifest["composite"]["rank"] == 4
    assert all(item["exact_rank_slice_audit"] for item in manifest["shape_audit"])
    packaged = json.loads(composition.training_result_path.read_text())
    assert packaged["schema"] == "jenny2.qlora-training.v1"
    assert packaged["status"] == "PASS"
    assert packaged["adapter_artifact_kind"] == "exact-rank-concatenated-skill-composite"
    assert packaged["adapter_model_sha256"] == composition.adapter_model_sha256
    assert packaged["adapter_config_sha256"] == composition.adapter_config_sha256
    assert packaged["rank"] == 4
    assert packaged["trained_residual_rank"] == 2
    assert packaged["composition_manifest_ref"] == composition.manifest_ref


def test_composition_outputs_are_content_deterministic(tmp_path: Path) -> None:
    parent = _adapter(tmp_path, "initial", offset=0.1)
    reading = _adapter(tmp_path, "reading", offset=0.2)
    result = _training_result(tmp_path, reading)
    kwargs = {
        "components": [parent, reading],
        "training_result_source": result,
        "expected_training_result_sha256": _sha(result),
    }
    left = compose_skill_adapters(**kwargs, output_root=tmp_path / "left")
    right = compose_skill_adapters(**kwargs, output_root=tmp_path / "right")
    assert left.adapter_model_sha256 == right.adapter_model_sha256
    assert left.adapter_config_sha256 == right.adapter_config_sha256
    assert left.manifest_ref == right.manifest_ref
    assert left.manifest_sha256 == right.manifest_sha256


def test_hash_or_training_lineage_failure_writes_nothing(tmp_path: Path) -> None:
    parent = _adapter(tmp_path, "initial", offset=0.1)
    reading = _adapter(tmp_path, "reading", offset=0.2)
    result = _training_result(tmp_path, reading)
    forged = SkillAdapterComponent(
        name=parent.name,
        path=parent.path,
        adapter_model_sha256="0" * 64,
        adapter_config_sha256=parent.adapter_config_sha256,
    )
    output = tmp_path / "compiled"
    with pytest.raises(SkillAdapterCompositionError, match="model SHA-256 differs"):
        compose_skill_adapters(
            [forged, reading],
            training_result_source=result,
            expected_training_result_sha256=_sha(result),
            output_root=output,
        )
    assert not output.exists()

    with pytest.raises(SkillAdapterCompositionError, match="training result SHA-256 differs"):
        compose_skill_adapters(
            [parent, reading],
            training_result_source=result,
            expected_training_result_sha256="f" * 64,
            output_root=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("difference", ["base", "targets", "nonfinite"])
def test_incompatible_components_fail_closed(tmp_path: Path, difference: str) -> None:
    parent = _adapter(tmp_path, "initial", offset=0.1)
    reading = _adapter(
        tmp_path,
        "reading",
        offset=0.2,
        base="/models/other" if difference == "base" else "/models/frozen-qwen",
        targets=["q_proj"] if difference == "targets" else None,
        nonfinite=difference == "nonfinite",
    )
    result = _training_result(tmp_path, reading)
    output = tmp_path / "compiled"
    with pytest.raises(SkillAdapterCompositionError):
        compose_skill_adapters(
            [parent, reading],
            training_result_source=result,
            expected_training_result_sha256=_sha(result),
            output_root=output,
        )
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    parent = _adapter(tmp_path, "initial", offset=0.1)
    reading = _adapter(tmp_path, "reading", offset=0.2)
    result = _training_result(tmp_path, reading)
    output = tmp_path / "compiled"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        compose_skill_adapters(
            [parent, reading],
            training_result_source=result,
            expected_training_result_sha256=_sha(result),
            output_root=output,
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
