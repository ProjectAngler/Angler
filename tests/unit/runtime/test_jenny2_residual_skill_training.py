from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "train_jenny2_temporal_qlora.py"
)
SPEC = importlib.util.spec_from_file_location(
    "jenny2_residual_skill_training_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeTuner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foundation = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict(
            {
                "prior_competence": nn.Linear(2, 2, bias=False),
                "library_reading": nn.Linear(2, 2, bias=False),
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                "prior_competence": nn.Linear(2, 2, bias=False),
                "library_reading": nn.Linear(2, 2, bias=False),
            }
        )
        self.activation_calls: list[tuple[object, bool]] = []
        self.active: tuple[str, ...] = ("prior_competence", "library_reading")

    def set_adapter(self, names, *, inference_mode: bool = False) -> None:
        self.activation_calls.append((names, inference_mode))
        if isinstance(names, str):
            self.active = (names,)
        else:
            self.active = tuple(names)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                not inference_mode and any(item in name.split(".") for item in self.active)
            )


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _FakeTuner()
        self._adapters_disabled = False

    @contextmanager
    def disable_adapter(self):
        previous = self._adapters_disabled
        self._adapters_disabled = True
        try:
            yield
        finally:
            self._adapters_disabled = previous


def test_residual_trainability_freezes_parent_and_foundation() -> None:
    model = _FakeModel()
    MODULE._activate_adapter_components(
        model,
        ("prior_competence", "library_reading"),
        inference_mode=False,
    )
    audit = MODULE._enforce_residual_skill_trainability(
        model,
        parent_adapter_name="prior_competence",
        skill_adapter_name="library_reading",
    )

    assert model.base_model.activation_calls[-1] == (
        ["prior_competence", "library_reading"],
        False,
    )
    assert audit["parent_trainable_tensor_count"] == 0
    assert audit["skill_trainable_tensor_count"] == audit["skill_tensor_count"]
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is ("library_reading" in name.split("."))


def test_parent_fingerprint_is_independent_of_skill_updates() -> None:
    model = _FakeModel()
    before = MODULE._adapter_parameter_fingerprint(model, "prior_competence")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "library_reading" in name.split("."):
                parameter.add_(1.0)
    after = MODULE._adapter_parameter_fingerprint(model, "prior_competence")
    assert after == before


def test_exact_adapter_identity_fails_closed(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"model")
    (adapter / "adapter_config.json").write_bytes(b"config")
    model_sha = MODULE._sha(adapter / "adapter_model.safetensors")
    config_sha = MODULE._sha(adapter / "adapter_config.json")
    assert MODULE._validate_exact_local_adapter(
        adapter,
        expected_model_sha256=model_sha,
        expected_config_sha256=config_sha,
    ) == (model_sha, config_sha)
    with pytest.raises(ValueError, match="tensor SHA-256 differs"):
        MODULE._validate_exact_local_adapter(
            adapter,
            expected_model_sha256="0" * 64,
            expected_config_sha256=config_sha,
        )


class _SavingModel:
    def save_pretrained(self, root: Path, **kwargs) -> None:
        selected = kwargs["selected_adapters"]
        (root / selected[0]).mkdir(parents=True)
        (root / "README.md").write_text("card", encoding="utf-8")
        (root / selected[0] / "adapter_model.safetensors").write_bytes(b"skill")
        (root / selected[0] / "adapter_config.json").write_text(
            "{}", encoding="utf-8"
        )


def test_selected_skill_is_saved_as_standalone_adapter_root(tmp_path: Path) -> None:
    path = MODULE._save_selected_adapter(
        _SavingModel(), tmp_path / "adapter", "library_reading"
    )
    assert path == tmp_path / "adapter"
    assert (path / "adapter_model.safetensors").read_bytes() == b"skill"
    assert (path / "adapter_config.json").is_file()
    assert not (path / "library_reading").exists()


class _EvalModel(_FakeModel):
    def forward(self, **batch):
        if self._adapters_disabled:
            key = "base"
        else:
            key = "+".join(self.base_model.active)
        losses = {
            "base": 4.0,
            "prior_competence": 3.0,
            "library_reading": 2.0,
            "prior_competence+library_reading": 1.0,
        }
        target = batch["shift_labels"]
        logits = torch.zeros((*target.shape, 3), dtype=torch.float32)
        logits[..., 1] = 1.0
        return SimpleNamespace(loss=torch.tensor(losses[key]), logits=logits)


def test_four_way_adapter_removal_evaluation_uses_exact_components(monkeypatch) -> None:
    model = _EvalModel()
    encoded = (
        torch.tensor([[0, 1, 1]]),
        torch.ones((1, 3), dtype=torch.long),
        torch.tensor([[-100, 1, 1]]),
    )
    monkeypatch.setattr(MODULE, "_encode", lambda *_args, **_kwargs: encoded)
    result = MODULE._evaluate_adapter_variants(
        model=model,
        tokenizer=object(),
        rows=[{"skill": "reading"}],
        maximum_tokens=32,
        input_device=torch.device("cpu"),
        variants={
            "base": (),
            "parent_only": ("prior_competence",),
            "skill_only": ("library_reading",),
            "parent_plus_skill": ("prior_competence", "library_reading"),
        },
        restore_adapters=("prior_competence", "library_reading"),
        trainable_adapter=None,
    )
    assert result["variants"]["base"]["mean_loss"] == 4.0
    assert result["variants"]["parent_only"]["mean_loss"] == 3.0
    assert result["variants"]["skill_only"]["mean_loss"] == 2.0
    assert result["variants"]["parent_plus_skill"]["mean_loss"] == 1.0
    assert result["variant_components"]["parent_plus_skill"] == [
        "prior_competence",
        "library_reading",
    ]
