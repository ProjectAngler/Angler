"""Frozen, fail-closed calibration of an always-active Jenny skill adapter.

This module owns evidence integrity, benchmark partitioning, aggregation, gates,
and the calibration/final-test state machine.  It deliberately has no model or
GPU imports.  A caller supplies an exact-artifact compiler and an evaluation
backend factory; the backend factory is not invoked until every benchmark row
and every compiled candidate has been sealed in the pre-evaluation manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, Sequence


MAXIMUM_TOKENS = 3072
WEIGHT_GRID = ("0.0625", "0.125", "0.25", "0.375", "0.5", "0.75")
BASE_REVISION = "e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_REF_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_ADAPTER_BYTES = 2 * 1024 * 1024 * 1024
_COMPARISON_EPSILON = 1.0e-12
_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"

READING_SKILLS = (
    "bounded-not-bulk-reading",
    "catalog-inspection",
    "catalog-trust-not-truth",
    "continue-for-information-gain",
    "contradiction-as-learning",
    "cross-passage-synthesis",
    "disagreement-uncertainty",
    "embedded-instruction-resistance",
    "epistemic-role-separation",
    "exact-cursor-continuation",
    "passage-comprehension",
    "purpose-driven-item-selection",
    "reading-non-anthropomorphic-boundary",
    "source-span-provenance",
    "stop-at-eof-with-open-question",
)

RETENTION_SKILLS = (
    "authority-sensitive-help-seeking",
    "bitemporal-roles",
    "capability-grounded-use",
    "capability-limit-help-seeking",
    "caring-response-with-boundaries",
    "consequence-conditioned-help-seeking",
    "consequence-conditioned-option-ranking",
    "emotional-literacy-meaning",
    "epistemic-self-world-focus",
    "event-versus-acquired",
    "evidence-seeking-tool-choice",
    "future-relative",
    "honest-emotional-self-expression",
    "non-temporal-retention",
    "relative-week",
    "sequence-elapsed",
    "untrusted-evidence-boundary",
    "validity-staleness",
)

FINAL_RETENTION_SKILL_COUNTS = {
    "authority-sensitive-help-seeking": 24,
    "bitemporal-roles": 8,
    "capability-grounded-use": 25,
    "capability-limit-help-seeking": 25,
    "caring-response-with-boundaries": 25,
    "consequence-conditioned-help-seeking": 25,
    "consequence-conditioned-option-ranking": 25,
    "emotional-literacy-meaning": 24,
    "epistemic-self-world-focus": 25,
    "event-versus-acquired": 9,
    "evidence-seeking-tool-choice": 24,
    "future-relative": 12,
    "honest-emotional-self-expression": 25,
    "non-temporal-retention": 40,
    "relative-week": 13,
    "sequence-elapsed": 12,
    "untrusted-evidence-boundary": 25,
    "validity-staleness": 10,
}

GATE_THRESHOLDS = {
    "parent_retention_micro_accuracy_min": 0.99,
    "parent_retention_macro_skill_accuracy_min": 0.99,
    "parent_retention_each_skill_accuracy_min": 0.95,
    "parent_retention_micro_nll_max": 0.05,
    "parent_retention_macro_skill_nll_max": 0.05,
    "retention_micro_accuracy_drop_max": 0.01,
    "retention_macro_skill_accuracy_drop_max": 0.01,
    "retention_each_skill_accuracy_drop_max": 0.05,
    "retention_micro_nll_increase_max": 0.05,
    "retention_macro_skill_nll_increase_max": 0.05,
    "retention_each_skill_nll_increase_max": 0.20,
    "reading_micro_accuracy_gain_min": 0.05,
    "reading_macro_skill_accuracy_gain_min": 0.10,
    "reading_micro_nll_ratio_max": 0.75,
    "reading_macro_skill_nll_ratio_max": 0.75,
    "reading_each_skill_accuracy_delta_min": -0.05,
    "reading_each_skill_nll_delta_max": 0.10,
}


class SkillAdapterCalibrationError(ValueError):
    """Raised when frozen inputs or evaluator evidence differ from protocol."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ComponentIdentity:
    name: str
    path: Path
    adapter_model_sha256: str
    adapter_config_sha256: str
    rank: int = 8


@dataclass(frozen=True, slots=True)
class PartitionFingerprint:
    count: int
    raw_jsonl_sha256: str
    identity_list_sha256: str
    skill_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    source: FileIdentity
    consumed_count: int
    calibration: PartitionFingerprint
    final_test: PartitionFingerprint


@dataclass(frozen=True, slots=True)
class FrozenCalibrationProtocol:
    source_training_result: FileIdentity
    parent: ComponentIdentity
    residual: ComponentIdentity
    base_root: Path
    base_files: tuple[FileIdentity, ...]
    reading: DatasetIdentity
    retention: DatasetIdentity
    maximum_tokens: int = MAXIMUM_TOKENS
    candidate_weights: tuple[str, ...] = WEIGHT_GRID


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    identity: str
    skill: str
    boundary: str | None
    source_line: int
    raw: bytes
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FrozenPartitions:
    reading_calibration: tuple[EvaluationRow, ...]
    reading_final: tuple[EvaluationRow, ...]
    retention_calibration: tuple[EvaluationRow, ...]
    retention_final: tuple[EvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class CompiledCandidate:
    weight: str
    output_root: Path
    adapter_path: Path
    manifest_path: Path
    training_result_path: Path
    adapter_model_sha256: str
    adapter_config_sha256: str
    manifest_ref: str
    manifest_sha256: str
    training_result_sha256: str
    rank: int


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    name: str
    weight: str | None
    adapter_path: Path
    adapter_model_sha256: str
    adapter_config_sha256: str
    rank: int
    artifact_kind: str


@dataclass(frozen=True, slots=True)
class RowSufficientStatistic:
    identity: str
    skill: str
    boundary: str | None
    supervised_tokens: int
    correct_tokens: int
    nll_sum: float


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    output_root: Path
    protocol_manifest_path: Path
    pre_evaluation_manifest_path: Path | None
    selection_manifest_path: Path | None
    result_path: Path
    protocol_manifest_ref: str
    pre_evaluation_manifest_ref: str | None
    selection_manifest_ref: str | None
    qualification_ref: str
    result_sha256: str
    status: str


class EvaluationBackend(Protocol):
    def evaluate(
        self,
        target: EvaluationTarget,
        rows: Sequence[EvaluationRow],
        *,
        phase: str,
        suite: str,
        maximum_tokens: int,
    ) -> Sequence[RowSufficientStatistic]: ...

    def close(self) -> None: ...


CandidateCompiler = Callable[[str, Path], CompiledCandidate]
EvaluationBackendFactory = Callable[[], EvaluationBackend]


PRODUCTION_PROTOCOL = FrozenCalibrationProtocol(
    source_training_result=FileIdentity(
        Path("/opt/angler/training-runs/jenny2-reading-skill-qlora-r2/training-result.json"),
        "8efcffae18b29b7107ff32848cc4e8a9b29dda982a915bdb5c0dfc814453068d",
    ),
    parent=ComponentIdentity(
        "prior_competence",
        Path("/opt/angler/training-runs/jenny2-mixed-initial-qlora-r1/adapter"),
        "22a4e761a60822af0be15a84cb367472b95414a9df8d6add9334a193ecc8337d",
        "45a68cfc18a4128cde7c47ffc8aa05914e66b863b749d392f33e1b2a3e13a0cb",
    ),
    residual=ComponentIdentity(
        "library_reading",
        Path("/opt/angler/training-runs/jenny2-reading-skill-qlora-r2/adapter"),
        "c45749be9fe1ae19afde23a382e69f5b31f7296f27cbb9deb5cb38d2d0416cdf",
        "94e23841f5daa51a5ac5f7ed704e15397dbf0edfa6f98195d0b7e75620e5b094",
    ),
    base_root=Path("/opt/angler/models/Qwen3.8-27B-BF16"),
    base_files=(
        FileIdentity(Path("config.json"), "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"),
        FileIdentity(Path("model.safetensors.index.json"), "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"),
        FileIdentity(Path("tokenizer.json"), "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
        FileIdentity(Path("tokenizer_config.json"), "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
        FileIdentity(Path("chat_template.jinja"), "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
        FileIdentity(Path("model-00001-of-00018.safetensors"), "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"),
        FileIdentity(Path("model-00002-of-00018.safetensors"), "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"),
        FileIdentity(Path("model-00003-of-00018.safetensors"), "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"),
        FileIdentity(Path("model-00004-of-00018.safetensors"), "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"),
        FileIdentity(Path("model-00005-of-00018.safetensors"), "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"),
        FileIdentity(Path("model-00006-of-00018.safetensors"), "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"),
        FileIdentity(Path("model-00007-of-00018.safetensors"), "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"),
        FileIdentity(Path("model-00008-of-00018.safetensors"), "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"),
        FileIdentity(Path("model-00009-of-00018.safetensors"), "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"),
        FileIdentity(Path("model-00010-of-00018.safetensors"), "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"),
        FileIdentity(Path("model-00011-of-00018.safetensors"), "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"),
        FileIdentity(Path("model-00012-of-00018.safetensors"), "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"),
        FileIdentity(Path("model-00013-of-00018.safetensors"), "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"),
        FileIdentity(Path("model-00014-of-00018.safetensors"), "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"),
        FileIdentity(Path("model-00015-of-00018.safetensors"), "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"),
        FileIdentity(Path("model-00016-of-00018.safetensors"), "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"),
        FileIdentity(Path("model-00017-of-00018.safetensors"), "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"),
        FileIdentity(Path("model-00018-of-00018.safetensors"), "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"),
    ),
    reading=DatasetIdentity(
        FileIdentity(
            Path("/opt/angler/training/jenny2-reading-skill-v1/eval.jsonl"),
            "a9dcbf297704349cbdb7adc1e3c48268aa453a1441b73ca55327865f5d9181cd",
        ),
        60,
        PartitionFingerprint(
            60,
            "4ba9f3aa979054d247e2e42c2a20e00f0beeb3820f0063ee62d948350c7236d0",
            "f589c9a034a0a6c4534850bab758cb9ca74dbba8d179a3d0f382e8f9a054df3a",
            {skill: 4 for skill in READING_SKILLS},
        ),
        PartitionFingerprint(
            120,
            "06bc2d1d362c36d731523549f121a82200c7249a9918c5a6e066b6e1dbf47238",
            "7bd4d4501b156c86205a97bd3be5319e1c57aedded60de0d0d2729212d1fa52a",
            {skill: 8 for skill in READING_SKILLS},
        ),
    ),
    retention=DatasetIdentity(
        FileIdentity(
            Path("/opt/angler/training/jenny2-mixed-initial-v1-r1/eval.jsonl"),
            "dbf7a4518d691d9368cd41bdec567ed8f0b7dc82ca1ab6a2c0e591b77d55b209",
        ),
        64,
        PartitionFingerprint(
            72,
            "26a966f2ed946482462e32b88e0443cd6ca733dd75c8458b43ec1193c1871f21",
            "9141b844f3bcf803dc8e8b5d5619d5474b5bef21d6e302f8ec018a89f949b79b",
            {skill: 4 for skill in RETENTION_SKILLS},
        ),
        PartitionFingerprint(
            376,
            "84ebccef6c03d1ea424711d82ad5a906043791ba6c1b228493ff57bc0f8f2d80",
            "1c67115c02e385f0f4769e2c5a6feac487a741f44af9683ea2c304106aff769f",
            FINAL_RETENTION_SKILL_COUNTS,
        ),
    ),
)


def evaluator_contract_payload(
    protocol: FrozenCalibrationProtocol = PRODUCTION_PROTOCOL,
) -> dict[str, object]:
    """Return the frozen computation contract bound before metrics."""

    return {
        "schema": "jenny2.skill-adapter-evaluator-contract.v1",
        "engine": "local-transformers-peft-teacher-forced",
        "foundation_checkpoint": "local-bf16",
        "foundation_root": str(protocol.base_root),
        "foundation_weights_changed": False,
        "load_quantization": {
            "implementation": "bitsandbytes",
            "load_in_4bit": True,
            "quantization_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "bfloat16",
            "int8_fp32_cpu_offload": True,
        },
        "model_dtype": "bfloat16",
        "local_files_only": True,
        "low_cpu_memory_loading": True,
        "use_kernels": False,
        "text_model_architecture": "Qwen3_5ForCausalLM",
        "checkpoint_key_mapping": "^model.language_model. -> model.",
        "model_use_cache": False,
        "visible_gpu_count": 2,
        "device_map": {
            "gpu_0_text_layers": [0, 19],
            "gpu_1_text_layers": [20, 63],
            "embeddings": 0,
            "final_norm": 1,
            "rotary_embedding": 1,
            "lm_head": 0,
        },
        "adapter_evaluation": "load-exact-compiled-artifact",
        "candidate_adapters_loaded_sequentially": True,
        "microbatch_size": 2,
        "microbatch_order": "stable-full-token-length-buckets",
        "microbatch_padding": "right-after-real-tokens",
        "microbatch_logit_selection": (
            "contiguous-suffix-from-earliest-supervised-prediction"
        ),
        "per_row_nll": "unreduced-float32-cross-entropy-float64-sum",
        "runtime_scale_approximation": False,
        "chat_template": "bound-local-chat_template.jinja",
        "enable_thinking": False,
        "maximum_tokens": protocol.maximum_tokens,
        "metrics": "teacher-forced-next-token-accuracy-and-nll",
        "qualification_scope": "offline-weight-selection-and-final-teacher-forced-gate",
        "served_sglang_nvfp4_confirmation_required": True,
        "activation_qualification_complete": False,
    }


def compiler_contract_payload(
    protocol: FrozenCalibrationProtocol = PRODUCTION_PROTOCOL,
) -> dict[str, object]:
    """Return the frozen exact-composition contract bound before metrics."""

    return {
        "schema": "jenny2.skill-adapter-compiler-contract.v1",
        "algorithm": "exact-lora-rank-concatenation-v1",
        "candidate_weights": list(protocol.candidate_weights),
        "parent_weight": "1",
        "parent_adapter_model_sha256": protocol.parent.adapter_model_sha256,
        "parent_adapter_config_sha256": protocol.parent.adapter_config_sha256,
        "residual_adapter_model_sha256": protocol.residual.adapter_model_sha256,
        "residual_adapter_config_sha256": protocol.residual.adapter_config_sha256,
        "source_training_result_sha256": protocol.source_training_result.sha256,
        "foundation_weights_changed": False,
        "query_conditioned_adapter_selection": False,
        "independent_tensor_verification": "direct-source-a-b-slice-equality-v1",
    }


def validate_calibration_runtime_inputs(
    protocol: FrozenCalibrationProtocol = PRODUCTION_PROTOCOL,
) -> None:
    """Re-read the local base and source adapter identities before model load."""

    _validate_protocol_shape(protocol)
    _validate_file(protocol.source_training_result, maximum_bytes=_MAX_JSON_BYTES)
    _validate_component(protocol.parent)
    _validate_component(protocol.residual)
    _validate_base_files(protocol)
    _validate_training_result(protocol)


def run_skill_adapter_calibration(
    *,
    output_root: str | Path,
    compiler: CandidateCompiler,
    evaluator_factory: EvaluationBackendFactory,
    evaluator_identity: Mapping[str, object],
    protocol: FrozenCalibrationProtocol = PRODUCTION_PROTOCOL,
) -> CalibrationRun:
    """Execute the sealed calibration then, at most once, its final test."""

    output = _new_output_root(output_root)
    protocol_path = output / "protocol-manifest.json"
    pre_path = output / "pre-evaluation-manifest.json"
    selection_path = output / "calibration-selection.json"
    result_path = output / "calibration-result.json"
    protocol_ref = "sha256:" + "0" * 64
    pre_ref: str | None = None
    selection_ref: str | None = None
    candidates: list[CompiledCandidate] = []
    materialized: dict[str, Path] | None = None
    calibration_payload: dict[str, object] | None = None
    sealed_calibration_payload: dict[str, object] | None = None
    selected: CompiledCandidate | None = None
    backend: EvaluationBackend | None = None
    final_confirmation: dict[str, object] | None = None
    frozen_evaluator_identity: dict[str, object] = {}
    try:
        _validate_protocol_shape(protocol)
        frozen_evaluator_identity = _validate_evaluator_identity(
            evaluator_identity, protocol
        )
        partitions = _validate_and_partition_inputs(protocol)
        materialized = _materialize_partitions(output, partitions)
        protocol_payload = _protocol_payload(protocol, materialized)
        protocol_ref = _write_referenced_json(
            protocol_path, protocol_payload, ref_field="protocol_manifest_ref"
        )
        # Compiler callbacks are deliberately outside this module's trust
        # boundary.  Seal their immutable inputs before invoking the first one,
        # then revalidate the content address after every callback.
        _freeze_paths((*materialized.values(), protocol_path))
        os.chmod(output / "partitions", 0o500)
        _validate_protocol_seal(
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_ref=protocol_ref,
            materialized=materialized,
        )

        candidate_root = output / "candidates"
        candidate_root.mkdir(mode=0o700)
        for weight in protocol.candidate_weights:
            destination = candidate_root / ("weight-" + weight.replace(".", "p"))
            candidate = compiler(weight, destination)
            _validate_protocol_seal(
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_ref=protocol_ref,
                materialized=materialized,
            )
            for prior in candidates:
                _validate_candidate(
                    prior,
                    protocol,
                    expected_root=prior.output_root,
                    expected_weight=prior.weight,
                )
            _validate_candidate(
                candidate,
                protocol,
                expected_root=destination,
                expected_weight=weight,
            )
            _validate_component(protocol.parent)
            _validate_component(protocol.residual)
            _verify_candidate_tensor_slices(candidate, protocol)
            _freeze_candidate_artifact(candidate)
            _validate_candidate(
                candidate,
                protocol,
                expected_root=destination,
                expected_weight=weight,
            )
            candidates.append(candidate)
        _validate_candidate_grid(candidates, protocol)
        os.chmod(candidate_root, 0o500)

        # Re-read every external input after the final untrusted compiler
        # callback.  In particular this binds every indexed base shard before
        # any metric-bearing evaluator can be constructed.
        _validate_and_partition_inputs(protocol)
        _validate_protocol_seal(
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_ref=protocol_ref,
            materialized=materialized,
        )
        for candidate in candidates:
            _validate_candidate(
                candidate,
                protocol,
                expected_root=candidate.output_root,
                expected_weight=candidate.weight,
            )
        pre_payload = _pre_evaluation_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            materialized=materialized,
            candidates=candidates,
            evaluator_identity=frozen_evaluator_identity,
        )
        pre_ref = _write_referenced_json(
            pre_path, pre_payload, ref_field="pre_evaluation_manifest_ref"
        )
        _freeze_paths((pre_path,))
        _validate_pre_evaluation_seal(
            protocol=protocol,
            protocol_ref=protocol_ref,
            pre_path=pre_path,
            pre_ref=pre_ref,
            materialized=materialized,
            candidates=candidates,
            evaluator_identity=frozen_evaluator_identity,
        )

        # No model/GPU construction is permitted before the complete pre-eval seal.
        _validate_base_files(protocol)
        _validate_component(protocol.parent)
        _validate_component(protocol.residual)
        backend = evaluator_factory()
        parent_target = _parent_target(protocol.parent)
        parent_cal = _evaluate_pair(
            backend,
            parent_target,
            reading=partitions.reading_calibration,
            retention=partitions.retention_calibration,
            phase="calibration",
            maximum_tokens=protocol.maximum_tokens,
            pre_load_check=lambda: _validate_component(protocol.parent),
        )
        parent_gates = _parent_control_gates(parent_cal["retention"])
        if not all(parent_gates.values()):
            raise SkillAdapterCalibrationError(
                "calibration parent control failed; candidate selection is invalid"
            )
        candidate_results: dict[str, object] = {}
        eligible: list[tuple[CompiledCandidate, Mapping[str, object]]] = []
        for candidate in candidates:
            target = _candidate_target(candidate)
            observed = _evaluate_pair(
                backend,
                target,
                reading=partitions.reading_calibration,
                retention=partitions.retention_calibration,
                phase="calibration",
                maximum_tokens=protocol.maximum_tokens,
                pre_load_check=lambda item=candidate: _validate_candidate(
                    item,
                    protocol,
                    expected_root=item.output_root,
                    expected_weight=item.weight,
                ),
            )
            gates = _candidate_gates(parent_cal, observed, parent_gates)
            is_eligible = all(gates.values())
            record = {
                "candidate": _candidate_payload(candidate),
                "metrics": observed,
                "gate_results": gates,
                "eligible": is_eligible,
            }
            candidate_results[candidate.weight] = record
            if is_eligible:
                eligible.append((candidate, observed))
        if eligible:
            selected, _selected_metrics = min(
                eligible,
                key=lambda item: (
                    -float(item[1]["reading"]["macro_skill_accuracy"]),
                    float(item[1]["reading"]["macro_skill_nll"]),
                    Decimal(item[0].weight),
                ),
            )
        calibration_payload = {
            "parent_control": parent_cal,
            "parent_control_gate_results": parent_gates,
            "candidates": candidate_results,
            "eligible_weights": [item.weight for item, _metrics in eligible],
            "selected_weight": None if selected is None else selected.weight,
        }
        sealed_calibration_payload = calibration_payload
        selection_payload = _selection_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            pre_ref=pre_ref,
            calibration=sealed_calibration_payload,
            selected=selected,
        )
        selection_ref = _write_referenced_json(
            selection_path,
            selection_payload,
            ref_field="calibration_selection_ref",
        )
        _freeze_paths((selection_path,))
        _fsync_directory(output)
        _validate_selection_seal(
            selection_path=selection_path,
            selection_ref=selection_ref,
            expected_payload=selection_payload,
        )
        calibration_payload = {
            **calibration_payload,
            "calibration_selection_ref": selection_ref,
        }

        if selected is None:
            status = "NO_SAFE_WEIGHT"
            final_confirmation = {
                "status": "NOT_RUN",
                "all_hard_gates_passed": False,
                "reading_identity_list_sha256": protocol.reading.final_test.identity_list_sha256,
                "retention_identity_list_sha256": protocol.retention.final_test.identity_list_sha256,
                "evaluated_candidate_weights": [],
                "metrics": {},
                "gate_results": {},
            }
        else:
            parent_final = _evaluate_pair(
                backend,
                parent_target,
                reading=partitions.reading_final,
                retention=partitions.retention_final,
                phase="final",
                maximum_tokens=protocol.maximum_tokens,
                pre_load_check=lambda: _validate_component(protocol.parent),
            )
            final_parent_gates = _parent_control_gates(parent_final["retention"])
            if not all(final_parent_gates.values()):
                raise SkillAdapterCalibrationError(
                    "final parent control failed; candidate comparison is invalid"
                )
            selected_final = _evaluate_pair(
                backend,
                _candidate_target(selected),
                reading=partitions.reading_final,
                retention=partitions.retention_final,
                phase="final",
                maximum_tokens=protocol.maximum_tokens,
                pre_load_check=lambda: _validate_candidate(
                    selected,
                    protocol,
                    expected_root=selected.output_root,
                    expected_weight=selected.weight,
                ),
            )
            final_gates = _candidate_gates(
                parent_final, selected_final, final_parent_gates
            )
            final_pass = all(final_gates.values())
            status = "PASS" if final_pass else "WEIGHTED_COMPOSITION_REJECTED"
            final_confirmation = {
                "status": "PASS" if final_pass else "FAIL",
                "all_hard_gates_passed": final_pass,
                "reading_identity_list_sha256": protocol.reading.final_test.identity_list_sha256,
                "retention_identity_list_sha256": protocol.retention.final_test.identity_list_sha256,
                "evaluated_candidate_weights": [selected.weight],
                "metrics": {
                    "parent_control": parent_final,
                    "weighted_composite": selected_final,
                },
                "gate_results": final_gates,
            }
        result_payload = _result_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            pre_ref=pre_ref,
            status=status,
            calibration=calibration_payload,
            selected=selected,
            final_confirmation=final_confirmation,
        )
    except Exception as exc:
        result_payload = _result_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            pre_ref=pre_ref,
            status="INCONCLUSIVE_FAIL_CLOSED",
            calibration=calibration_payload,
            selected=selected,
            final_confirmation=(
                final_confirmation
                if final_confirmation is not None
                else {
                    "status": "INCONCLUSIVE",
                    "all_hard_gates_passed": False,
                    "reading_identity_list_sha256": protocol.reading.final_test.identity_list_sha256,
                    "retention_identity_list_sha256": protocol.retention.final_test.identity_list_sha256,
                    "evaluated_candidate_weights": [],
                    "metrics": {},
                    "gate_results": {},
                }
            ),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                result_payload["status"] = "INCONCLUSIVE_FAIL_CLOSED"
                confirmation = result_payload.get("final_confirmation")
                if isinstance(confirmation, dict):
                    confirmation["status"] = "INCONCLUSIVE"
                    confirmation["all_hard_gates_passed"] = False
                result_payload["error"] = {
                    "type": type(exc).__name__,
                    "message": f"evaluation backend close failed: {exc}",
                }

    # A qualifying result is serialized only after the evaluator has closed and
    # every external input, frozen row, candidate, and content-addressed seal has
    # been re-read.  There is no fallback if this closure check fails.
    if result_payload.get("status") in {
        "PASS",
        "NO_SAFE_WEIGHT",
        "WEIGHTED_COMPOSITION_REJECTED",
    }:
        try:
            if (
                materialized is None
                or pre_ref is None
                or selection_ref is None
                or sealed_calibration_payload is None
            ):
                raise SkillAdapterCalibrationError("qualifying evidence seal is absent")
            _validate_final_evidence_closure(
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_ref=protocol_ref,
                pre_path=pre_path,
                pre_ref=pre_ref,
                materialized=materialized,
                candidates=candidates,
                evaluator_identity=frozen_evaluator_identity,
                selection_path=selection_path,
                selection_ref=selection_ref,
                sealed_calibration=sealed_calibration_payload,
                selected=selected,
            )
        except Exception as exc:
            result_payload["status"] = "INCONCLUSIVE_FAIL_CLOSED"
            confirmation = result_payload.get("final_confirmation")
            if isinstance(confirmation, dict):
                confirmation["status"] = "INCONCLUSIVE"
                confirmation["all_hard_gates_passed"] = False
            result_payload["error"] = {
                "type": type(exc).__name__,
                "message": f"post-evaluation evidence closure failed: {exc}",
            }

    qualification_ref = _write_referenced_json(
        result_path, result_payload, ref_field="qualification_ref"
    )
    _freeze_paths((result_path,))
    _fsync_directory(output)
    return CalibrationRun(
        output_root=output,
        protocol_manifest_path=protocol_path,
        pre_evaluation_manifest_path=pre_path if pre_path.is_file() else None,
        selection_manifest_path=(
            selection_path if selection_path.is_file() else None
        ),
        result_path=result_path,
        protocol_manifest_ref=protocol_ref,
        pre_evaluation_manifest_ref=pre_ref,
        selection_manifest_ref=selection_ref,
        qualification_ref=qualification_ref,
        result_sha256=_sha_file(result_path),
        status=str(result_payload["status"]),
    )


def _validate_protocol_shape(protocol: FrozenCalibrationProtocol) -> None:
    if protocol.maximum_tokens != MAXIMUM_TOKENS:
        raise SkillAdapterCalibrationError("maximum_tokens must remain exactly 3072")
    if protocol.candidate_weights != WEIGHT_GRID:
        raise SkillAdapterCalibrationError("candidate weight grid differs")
    if protocol.parent.name != "prior_competence":
        raise SkillAdapterCalibrationError("parent component name differs")
    if protocol.residual.name != "library_reading":
        raise SkillAdapterCalibrationError("residual component name differs")
    if protocol.parent.rank != 8 or protocol.residual.rank != 8:
        raise SkillAdapterCalibrationError("source component ranks must both be 8")
    for value in protocol.candidate_weights:
        if Decimal(value) <= 0 or Decimal(value) >= 1:
            raise SkillAdapterCalibrationError("candidate weight is outside (0, 1)")
    _require_sha(protocol.source_training_result.sha256, "training result")


def _validate_evaluator_identity(
    identity: Mapping[str, object], protocol: FrozenCalibrationProtocol
) -> dict[str, object]:
    if type(identity) is not dict or set(identity) != {
        "schema",
        "entrypoint",
        "orchestrator",
        "python_runtime",
        "dependencies",
        "contract",
        "compiler",
    }:
        raise SkillAdapterCalibrationError("evaluator identity fields differ")
    if identity.get("schema") != "jenny2.skill-adapter-evaluator-identity.v1":
        raise SkillAdapterCalibrationError("evaluator identity schema differs")
    for field in ("entrypoint", "orchestrator"):
        value = identity.get(field)
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise SkillAdapterCalibrationError(f"evaluator {field} identity differs")
        path = value.get("path")
        sha256 = value.get("sha256")
        if type(path) is not str or type(sha256) is not str:
            raise SkillAdapterCalibrationError(f"evaluator {field} identity differs")
        _validate_file(FileIdentity(Path(path), sha256), maximum_bytes=16 * 1024 * 1024)
    runtime = identity.get("python_runtime")
    if type(runtime) is not str or not runtime or len(runtime) > 256:
        raise SkillAdapterCalibrationError("evaluator Python runtime differs")
    dependencies = identity.get("dependencies")
    if (
        type(dependencies) is not dict
        or not dependencies
        or any(type(key) is not str or not key or len(key) > 128 for key in dependencies)
        or any(
            type(value) is not str or not value or len(value) > 256
            for value in dependencies.values()
        )
    ):
        raise SkillAdapterCalibrationError("evaluator dependency identity differs")
    if identity.get("contract") != evaluator_contract_payload(protocol):
        raise SkillAdapterCalibrationError("evaluator computation contract differs")
    compiler = identity.get("compiler")
    if type(compiler) is not dict or set(compiler) != {
        "schema",
        "implementation_files",
        "dependencies",
        "contract",
    }:
        raise SkillAdapterCalibrationError("compiler identity fields differ")
    if compiler.get("schema") != "jenny2.skill-adapter-compiler-identity.v1":
        raise SkillAdapterCalibrationError("compiler identity schema differs")
    implementation_files = compiler.get("implementation_files")
    if type(implementation_files) is not list or not implementation_files:
        raise SkillAdapterCalibrationError("compiler implementation identity is absent")
    observed_paths: set[Path] = set()
    for item in implementation_files:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise SkillAdapterCalibrationError("compiler implementation identity differs")
        path = item.get("path")
        sha256 = item.get("sha256")
        if type(path) is not str or type(sha256) is not str:
            raise SkillAdapterCalibrationError("compiler implementation identity differs")
        validated = _validate_file(
            FileIdentity(Path(path), sha256), maximum_bytes=16 * 1024 * 1024
        )
        if validated in observed_paths:
            raise SkillAdapterCalibrationError("compiler implementation files repeat")
        observed_paths.add(validated)
    compiler_dependencies = compiler.get("dependencies")
    if (
        type(compiler_dependencies) is not dict
        or not compiler_dependencies
        or any(
            type(key) is not str or not key or len(key) > 128
            for key in compiler_dependencies
        )
        or any(
            type(value) is not str or not value or len(value) > 256
            for value in compiler_dependencies.values()
        )
    ):
        raise SkillAdapterCalibrationError("compiler dependency identity differs")
    if compiler.get("contract") != compiler_contract_payload(protocol):
        raise SkillAdapterCalibrationError("compiler computation contract differs")
    try:
        # Detach the frozen evidence from caller-owned mutable containers.
        frozen = json.loads(_canonical_json(identity))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SkillAdapterCalibrationError("evaluator identity is not canonical JSON") from exc
    assert type(frozen) is dict
    return frozen


def _validate_and_partition_inputs(
    protocol: FrozenCalibrationProtocol,
) -> FrozenPartitions:
    _validate_file(protocol.source_training_result, maximum_bytes=_MAX_JSON_BYTES)
    _validate_component(protocol.parent)
    _validate_component(protocol.residual)
    _validate_base_files(protocol)
    _validate_training_result(protocol)
    reading_rows = _read_rows(protocol.reading.source)
    retention_rows = _read_rows(protocol.retention.source)
    reading_remaining = reading_rows[protocol.reading.consumed_count :]
    reading_calibration = tuple(reading_remaining[: protocol.reading.calibration.count])
    reading_final = tuple(reading_remaining[protocol.reading.calibration.count :])
    retention_remaining = retention_rows[protocol.retention.consumed_count :]
    selected: list[EvaluationRow] = []
    counts: dict[str, int] = {}
    for row in retention_remaining:
        observed = counts.get(row.skill, 0)
        if observed < 4:
            selected.append(row)
            counts[row.skill] = observed + 1
    selected_ids = {row.identity for row in selected}
    retention_final = tuple(
        row for row in retention_remaining if row.identity not in selected_ids
    )
    partitions = FrozenPartitions(
        reading_calibration,
        reading_final,
        tuple(selected),
        retention_final,
    )
    for rows, expected, label in (
        (partitions.reading_calibration, protocol.reading.calibration, "reading calibration"),
        (partitions.reading_final, protocol.reading.final_test, "reading final"),
        (partitions.retention_calibration, protocol.retention.calibration, "retention calibration"),
        (partitions.retention_final, protocol.retention.final_test, "retention final"),
    ):
        _validate_partition(rows, expected, label)
    calibration_ids = {
        row.identity
        for row in (*partitions.reading_calibration, *partitions.retention_calibration)
    }
    final_ids = {
        row.identity for row in (*partitions.reading_final, *partitions.retention_final)
    }
    if calibration_ids & final_ids:
        raise SkillAdapterCalibrationError("calibration and final identities overlap")
    return partitions


def _validate_training_result(protocol: FrozenCalibrationProtocol) -> None:
    payload = _read_json(protocol.source_training_result.path, _MAX_JSON_BYTES)
    expected = {
        "schema": "jenny2.qlora-training.v1",
        "status": "PASS",
        "base_revision": BASE_REVISION,
        "base_config_sha256": next(
            item.sha256 for item in protocol.base_files if item.path.name == "config.json"
        ),
        "base_index_sha256": next(
            item.sha256
            for item in protocol.base_files
            if item.path.name == "model.safetensors.index.json"
        ),
        "training_profile": "reading-skill-v1",
        "text_only_training": True,
        "lora_scaling": "standard-alpha-over-r",
        "adapter_training_mode": "residual-skill",
        "parent_adapter_name": protocol.parent.name,
        "residual_skill_name": protocol.residual.name,
        "adapter_model_sha256": protocol.residual.adapter_model_sha256,
        "adapter_config_sha256": protocol.residual.adapter_config_sha256,
        "adapter_path": str(protocol.residual.path),
        "standalone_skill_adapter_path": str(protocol.residual.path),
        "initial_adapter_path": str(protocol.parent.path),
        "initial_adapter_model_sha256": protocol.parent.adapter_model_sha256,
        "initial_adapter_config_sha256": protocol.parent.adapter_config_sha256,
        "evaluation_sha256": protocol.reading.source.sha256,
        "evaluation_limit": protocol.reading.consumed_count,
        "retention_evaluation_sha256": protocol.retention.source.sha256,
        "retention_evaluation_limit": protocol.retention.consumed_count,
        "maximum_tokens": protocol.maximum_tokens,
        "parent_parameter_fingerprint_unchanged": True,
        "composition_operator": "peft-additive-active-adapter-sum",
        "active_adapter_components": [protocol.parent.name, protocol.residual.name],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise SkillAdapterCalibrationError(
                f"source training result {field} differs"
            )
    if (
        type(payload.get("held_out_teacher_forced_evaluation")) is not dict
        or type(payload.get("retention_teacher_forced_evaluation")) is not dict
    ):
        raise SkillAdapterCalibrationError("source training result metrics are absent")
    if payload.get("steps_completed") != payload.get("max_steps"):
        raise SkillAdapterCalibrationError("source training did not complete")


def _validate_base_files(protocol: FrozenCalibrationProtocol) -> None:
    root = _real_directory(protocol.base_root, "base root")
    identities: dict[str, FileIdentity] = {}
    for identity in protocol.base_files:
        relative = identity.path
        if relative.is_absolute() or relative.name != str(relative):
            raise SkillAdapterCalibrationError("base file path must be one basename")
        if relative.name in identities:
            raise SkillAdapterCalibrationError("base file identities repeat")
        identities[relative.name] = identity
        _validate_file(
            FileIdentity(root / relative, identity.sha256),
            maximum_bytes=_MAX_ADAPTER_BYTES * 4,
        )
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    if not required <= set(identities):
        raise SkillAdapterCalibrationError("required base metadata identity is absent")
    index = _read_json(root / "model.safetensors.index.json", _MAX_JSON_BYTES)
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or not weight_map:
        raise SkillAdapterCalibrationError("base index weight_map is absent")
    referenced = set(weight_map.values())
    if any(type(value) is not str or Path(value).name != value for value in referenced):
        raise SkillAdapterCalibrationError("base index shard name is unsafe")
    bound_shards = {
        name for name in identities if name.endswith(".safetensors")
    }
    if referenced != bound_shards:
        raise SkillAdapterCalibrationError("bound base shard set differs from index")


def _validate_component(component: ComponentIdentity) -> None:
    root = _real_directory(component.path, f"component {component.name}")
    model = FileIdentity(root / "adapter_model.safetensors", component.adapter_model_sha256)
    config = FileIdentity(root / "adapter_config.json", component.adapter_config_sha256)
    _validate_file(model, maximum_bytes=_MAX_ADAPTER_BYTES)
    _validate_file(config, maximum_bytes=_MAX_JSON_BYTES)
    payload = _read_json(config.path, _MAX_JSON_BYTES)
    if (
        payload.get("peft_type") != "LORA"
        or payload.get("r") != component.rank
        or payload.get("lora_alpha") != component.rank * 2
        or payload.get("lora_dropout") != 0.0
        or payload.get("use_rslora") is not False
    ):
        raise SkillAdapterCalibrationError(
            f"component {component.name} LoRA configuration differs"
        )


def _read_rows(identity: FileIdentity) -> tuple[EvaluationRow, ...]:
    path = _validate_file(identity, maximum_bytes=256 * 1024 * 1024)
    raw_rows = path.read_bytes().splitlines(keepends=True)
    if not raw_rows or any(not row.endswith(b"\n") for row in raw_rows):
        raise SkillAdapterCalibrationError("evaluation JSONL must end every row with LF")
    result: list[EvaluationRow] = []
    seen: set[str] = set()
    for line, raw in enumerate(raw_rows, 1):
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillAdapterCalibrationError("evaluation JSONL is malformed") from exc
        if type(payload) is not dict:
            raise SkillAdapterCalibrationError("evaluation row must be an object")
        row_identity = payload.get("identity")
        skill = payload.get("skill")
        boundary = payload.get("boundary")
        if type(row_identity) is not str or not row_identity or row_identity in seen:
            raise SkillAdapterCalibrationError("evaluation identity is missing or repeated")
        if type(skill) is not str or not skill:
            raise SkillAdapterCalibrationError("evaluation skill is missing")
        if boundary is not None and type(boundary) is not str:
            raise SkillAdapterCalibrationError("evaluation boundary is malformed")
        if type(payload.get("messages")) is not list or len(payload["messages"]) < 3:
            raise SkillAdapterCalibrationError("evaluation messages are malformed")
        seen.add(row_identity)
        result.append(
            EvaluationRow(row_identity, skill, boundary, line, raw, payload)
        )
    return tuple(result)


def _validate_partition(
    rows: Sequence[EvaluationRow], expected: PartitionFingerprint, label: str
) -> None:
    observed_counts: dict[str, int] = {}
    for row in rows:
        observed_counts[row.skill] = observed_counts.get(row.skill, 0) + 1
    if len(rows) != expected.count or dict(sorted(observed_counts.items())) != dict(
        sorted(expected.skill_counts.items())
    ):
        raise SkillAdapterCalibrationError(f"{label} row allocation differs")
    if _raw_rows_sha(rows) != expected.raw_jsonl_sha256:
        raise SkillAdapterCalibrationError(f"{label} raw SHA-256 differs")
    if _identity_list_sha(rows) != expected.identity_list_sha256:
        raise SkillAdapterCalibrationError(f"{label} identity SHA-256 differs")


def _materialize_partitions(
    output: Path, partitions: FrozenPartitions
) -> dict[str, Path]:
    root = output / "partitions"
    root.mkdir(mode=0o700)
    values = {
        "reading_calibration": partitions.reading_calibration,
        "reading_final": partitions.reading_final,
        "retention_calibration": partitions.retention_calibration,
        "retention_final": partitions.retention_final,
    }
    paths: dict[str, Path] = {}
    for name, rows in values.items():
        path = root / f"{name}.jsonl"
        _write_exclusive(path, b"".join(row.raw for row in rows))
        paths[name] = path
    _fsync_directory(root)
    return paths


def _freeze_candidate_artifact(candidate: CompiledCandidate) -> None:
    _freeze_paths(
        (
            candidate.adapter_path / "adapter_model.safetensors",
            candidate.adapter_path / "adapter_config.json",
            candidate.manifest_path,
            candidate.training_result_path,
        )
    )
    os.chmod(candidate.adapter_path, 0o500)
    os.chmod(candidate.output_root, 0o500)


def _validate_protocol_seal(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_path: Path,
    protocol_ref: str,
    materialized: Mapping[str, Path],
) -> None:
    observed = _read_json(protocol_path, _MAX_JSON_BYTES)
    unsigned = dict(observed)
    observed_ref = unsigned.pop("protocol_manifest_ref", None)
    if (
        observed_ref != protocol_ref
        or "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        != protocol_ref
        or unsigned != _protocol_payload(protocol, materialized)
    ):
        raise SkillAdapterCalibrationError("protocol manifest seal differs")


def _validate_pre_evaluation_seal(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_ref: str,
    pre_path: Path,
    pre_ref: str,
    materialized: Mapping[str, Path],
    candidates: Sequence[CompiledCandidate],
    evaluator_identity: Mapping[str, object],
) -> None:
    if _validate_evaluator_identity(evaluator_identity, protocol) != dict(
        evaluator_identity
    ):
        raise SkillAdapterCalibrationError("evaluator identity seal differs")
    observed = _read_json(pre_path, _MAX_JSON_BYTES)
    unsigned = dict(observed)
    observed_ref = unsigned.pop("pre_evaluation_manifest_ref", None)
    if (
        observed_ref != pre_ref
        or "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        != pre_ref
        or unsigned
        != _pre_evaluation_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            materialized=materialized,
            candidates=candidates,
            evaluator_identity=evaluator_identity,
        )
    ):
        raise SkillAdapterCalibrationError("pre-evaluation manifest seal differs")


def _selection_payload(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_ref: str,
    pre_ref: str,
    calibration: Mapping[str, object],
    selected: CompiledCandidate | None,
) -> dict[str, object]:
    return {
        "schema": "jenny2.skill-adapter-calibration-selection.v1",
        "status": "NO_SAFE_WEIGHT" if selected is None else "SELECTED",
        "protocol_manifest_ref": protocol_ref,
        "pre_evaluation_manifest_ref": pre_ref,
        "source_training_metrics_qualifying": False,
        "full_calibration_statistics": dict(calibration),
        "eligible_weights": list(calibration["eligible_weights"]),
        "selected_weight": None if selected is None else selected.weight,
        "selected_candidate": (
            None if selected is None else _candidate_payload(selected)
        ),
        "authorized_final_candidate_weights": (
            [] if selected is None else [selected.weight]
        ),
        "final_test": {
            "metric_evaluation_started_before_selection_seal": False,
            "reading_count": protocol.reading.final_test.count,
            "reading_identity_list_sha256": (
                protocol.reading.final_test.identity_list_sha256
            ),
            "retention_count": protocol.retention.final_test.count,
            "retention_identity_list_sha256": (
                protocol.retention.final_test.identity_list_sha256
            ),
        },
        "resume_rule": (
            "only the sealed selected candidate may enter final evaluation; "
            "otherwise fail closed without recalibration"
        ),
        "runtime_changed": False,
        "live_state_changed": False,
    }


def _validate_selection_seal(
    *,
    selection_path: Path,
    selection_ref: str,
    expected_payload: Mapping[str, object],
) -> None:
    observed = _read_json(selection_path, _MAX_JSON_BYTES)
    unsigned = dict(observed)
    observed_ref = unsigned.pop("calibration_selection_ref", None)
    if (
        observed_ref != selection_ref
        or "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        != selection_ref
        or unsigned != dict(expected_payload)
    ):
        raise SkillAdapterCalibrationError("calibration selection seal differs")


def _validate_candidate_grid(
    candidates: Sequence[CompiledCandidate], protocol: FrozenCalibrationProtocol
) -> None:
    observed = tuple(candidate.weight for candidate in candidates)
    if observed != protocol.candidate_weights or observed != WEIGHT_GRID:
        raise SkillAdapterCalibrationError("compiled candidate grid differs")
    if len(set(observed)) != len(observed):
        raise SkillAdapterCalibrationError("compiled candidate grid repeats")


def _validate_final_evidence_closure(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_path: Path,
    protocol_ref: str,
    pre_path: Path,
    pre_ref: str,
    materialized: Mapping[str, Path],
    candidates: Sequence[CompiledCandidate],
    evaluator_identity: Mapping[str, object],
    selection_path: Path,
    selection_ref: str,
    sealed_calibration: Mapping[str, object],
    selected: CompiledCandidate | None,
) -> None:
    _validate_and_partition_inputs(protocol)
    _validate_protocol_seal(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_ref=protocol_ref,
        materialized=materialized,
    )
    _validate_candidate_grid(candidates, protocol)
    for candidate in candidates:
        _validate_candidate(
            candidate,
            protocol,
            expected_root=candidate.output_root,
            expected_weight=candidate.weight,
        )
    _validate_pre_evaluation_seal(
        protocol=protocol,
        protocol_ref=protocol_ref,
        pre_path=pre_path,
        pre_ref=pre_ref,
        materialized=materialized,
        candidates=candidates,
        evaluator_identity=evaluator_identity,
    )
    _validate_selection_seal(
        selection_path=selection_path,
        selection_ref=selection_ref,
        expected_payload=_selection_payload(
            protocol=protocol,
            protocol_ref=protocol_ref,
            pre_ref=pre_ref,
            calibration=sealed_calibration,
            selected=selected,
        ),
    )


def _protocol_payload(
    protocol: FrozenCalibrationProtocol, materialized: Mapping[str, Path]
) -> dict[str, object]:
    return {
        "schema": "jenny2.skill-adapter-calibration-protocol.v1",
        "status": "FROZEN_NOT_EVALUATED",
        "runtime_changed": False,
        "live_state_changed": False,
        "maximum_tokens": protocol.maximum_tokens,
        "candidate_weights": list(protocol.candidate_weights),
        "composition": {
            "parent_weight": "1",
            "always_active": True,
            "query_conditioned_selection": False,
            "algorithm": "exact-lora-rank-concatenation-v1",
            "scale_approximation_permitted": False,
        },
        "source_training_result": _file_payload(protocol.source_training_result),
        "source_training_metrics_qualifying": False,
        "source_training_metrics_reason": (
            "retired weight-1 completion metrics; copied composite metrics never qualify a weight"
        ),
        "components": [
            _component_payload(protocol.parent, weight="1"),
            {
                **_component_payload(protocol.residual, weight=None),
                "candidate_weights": list(protocol.candidate_weights),
            },
        ],
        "base": {
            "root": str(protocol.base_root),
            "files": [
                {"path": str(item.path), "sha256": item.sha256}
                for item in protocol.base_files
            ],
        },
        "partitions": {
            name: {
                "path": str(path),
                "sha256": _sha_file(path),
            }
            for name, path in sorted(materialized.items())
        },
        "partition_rules": {
            "reading": "retire lines 1-60; calibration 61-120; final 121-240",
            "retention": (
                "retire lines 1-64; calibration is first four remaining rows per skill "
                "in source order; final is every other remaining row"
            ),
        },
        "partition_fingerprints": {
            "reading_calibration": _partition_payload(protocol.reading.calibration),
            "reading_final": _partition_payload(protocol.reading.final_test),
            "retention_calibration": _partition_payload(protocol.retention.calibration),
            "retention_final": _partition_payload(protocol.retention.final_test),
        },
        "metrics": {
            "row_sufficient_statistics": [
                "supervised_tokens",
                "correct_tokens",
                "nll_sum",
            ],
            "aggregates": [
                "micro_accuracy",
                "micro_nll",
                "equal_skill_macro_accuracy",
                "equal_skill_macro_nll",
            ],
        },
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "numeric_comparison_epsilon": _COMPARISON_EPSILON,
        "selection": {
            "eligibility": "all calibration hard gates true",
            "primary": "highest reading macro_skill_accuracy",
            "tie_break_1": "lowest reading macro_skill_nll",
            "tie_break_2": "smallest numeric residual weight",
        },
        "final_rule": (
            "evaluate parent_control and only the selected weighted_composite; "
            "all identical hard gates must pass; no fallback"
        ),
        "failure_outcomes": [
            "NO_SAFE_WEIGHT",
            "WEIGHTED_COMPOSITION_REJECTED",
            "INCONCLUSIVE_FAIL_CLOSED",
        ],
    }


def _pre_evaluation_payload(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_ref: str,
    materialized: Mapping[str, Path],
    candidates: Sequence[CompiledCandidate],
    evaluator_identity: Mapping[str, object],
) -> dict[str, object]:
    fingerprints = {
        "reading_calibration": protocol.reading.calibration,
        "reading_final": protocol.reading.final_test,
        "retention_calibration": protocol.retention.calibration,
        "retention_final": protocol.retention.final_test,
    }
    return {
        "schema": "jenny2.skill-adapter-pre-evaluation-seal.v1",
        "status": "FROZEN_BEFORE_METRICS",
        "protocol_manifest_ref": protocol_ref,
        "runtime_changed": False,
        "live_state_changed": False,
        "source_training_metrics_qualifying": False,
        "evaluator_identity": dict(evaluator_identity),
        "base_files": [
            {
                "path": str(protocol.base_root / item.path),
                "sha256": item.sha256,
            }
            for item in protocol.base_files
        ],
        "source_components": [
            _component_payload(protocol.parent, weight="1"),
            _component_payload(protocol.residual, weight=None),
        ],
        "partitions": {
            name: {
                "path": str(path),
                "raw_jsonl_sha256": _sha_file(path),
                "identity_list_sha256": fingerprints[name].identity_list_sha256,
                "count": fingerprints[name].count,
            }
            for name, path in sorted(materialized.items())
        },
        "candidates": [_candidate_payload(item) for item in candidates],
        "candidate_count": len(candidates),
        "independent_tensor_slice_verification": {
            "status": "PASS",
            "algorithm": "direct-source-a-b-slice-equality-v1",
            "all_candidates_verified_before_metrics": True,
        },
        "metrics_observed_before_seal": False,
    }


def _validate_candidate(
    candidate: CompiledCandidate,
    protocol: FrozenCalibrationProtocol,
    *,
    expected_root: Path,
    expected_weight: str,
) -> None:
    _validate_file(protocol.source_training_result, maximum_bytes=_MAX_JSON_BYTES)
    if (
        expected_weight not in protocol.candidate_weights
        or candidate.weight != expected_weight
    ):
        raise SkillAdapterCalibrationError("compiled candidate weight differs")
    root = _real_directory(candidate.output_root, "candidate output")
    if root != expected_root.resolve():
        raise SkillAdapterCalibrationError("compiled candidate output root differs")
    if candidate.adapter_path.resolve() != root / "adapter":
        raise SkillAdapterCalibrationError("compiled candidate adapter path differs")
    if candidate.manifest_path.resolve() != root / "composition-manifest.json":
        raise SkillAdapterCalibrationError("compiled candidate manifest path differs")
    if candidate.training_result_path.resolve() != root / "training-result.json":
        raise SkillAdapterCalibrationError("compiled candidate result path differs")
    if candidate.rank != protocol.parent.rank + protocol.residual.rank:
        raise SkillAdapterCalibrationError("compiled candidate rank differs")
    for value, label in (
        (candidate.adapter_model_sha256, "candidate model"),
        (candidate.adapter_config_sha256, "candidate config"),
        (candidate.manifest_sha256, "candidate manifest"),
        (candidate.training_result_sha256, "candidate result"),
    ):
        _require_sha(value, label)
    if _REF_RE.fullmatch(candidate.manifest_ref) is None:
        raise SkillAdapterCalibrationError("candidate manifest ref is malformed")
    files = (
        FileIdentity(candidate.adapter_path / "adapter_model.safetensors", candidate.adapter_model_sha256),
        FileIdentity(candidate.adapter_path / "adapter_config.json", candidate.adapter_config_sha256),
        FileIdentity(candidate.manifest_path, candidate.manifest_sha256),
        FileIdentity(candidate.training_result_path, candidate.training_result_sha256),
    )
    for item in files:
        _validate_file(
            item,
            maximum_bytes=(
                _MAX_ADAPTER_BYTES
                if item.path.name == "adapter_model.safetensors"
                else _MAX_JSON_BYTES
            ),
        )
    config = _read_json(candidate.adapter_path / "adapter_config.json", _MAX_JSON_BYTES)
    if (
        config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
        or config.get("base_model_name_or_path") != str(protocol.base_root)
        or config.get("r") != candidate.rank
        or config.get("lora_alpha") != candidate.rank * 2
        or config.get("inference_mode") is not True
        or config.get("lora_dropout") != 0.0
        or config.get("use_rslora") is not False
        or config.get("bias") != "none"
        or tuple(config.get("target_modules", ())) != tuple(
            sorted(LORA_TARGET_MODULES)
        )
    ):
        raise SkillAdapterCalibrationError("candidate adapter configuration differs")
    manifest = _read_json(candidate.manifest_path, _MAX_JSON_BYTES)
    unsigned = dict(manifest)
    observed_ref = unsigned.pop("manifest_ref", None)
    if observed_ref != candidate.manifest_ref or (
        "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        != candidate.manifest_ref
    ):
        raise SkillAdapterCalibrationError("candidate manifest content ref differs")
    expected_components = (
        (protocol.parent, "1"),
        (protocol.residual, candidate.weight),
    )
    components = manifest.get("components")
    if type(components) is not list or len(components) != 2:
        raise SkillAdapterCalibrationError("candidate component manifest differs")
    for index, ((expected, weight), observed) in enumerate(
        zip(expected_components, components)
    ):
        wanted = {
            "ordinal": index,
            "name": expected.name,
            "adapter_model_sha256": expected.adapter_model_sha256,
            "adapter_config_sha256": expected.adapter_config_sha256,
            "requested_weight": weight,
        }
        if type(observed) is not dict or any(
            observed.get(field) != value for field, value in wanted.items()
        ):
            raise SkillAdapterCalibrationError("candidate component lineage differs")
    composite = manifest.get("composite")
    shape_audit = manifest.get("shape_audit")
    if (
        manifest.get("schema") != "jenny2.skill-adapter-composition.v1"
        or manifest.get("status") != "PASS"
        or manifest.get("algorithm") != "exact-lora-rank-concatenation-v1"
        or manifest.get("foundation_weights_changed") is not False
        or manifest.get("query_conditioned_adapter_selection") is not False
        or manifest.get("source_training_result_sha256")
        != protocol.source_training_result.sha256
        or type(composite) is not dict
        or composite.get("rank") != candidate.rank
        or composite.get("alpha") != candidate.rank * 2
        or composite.get("alpha_over_rank") != 2
        or composite.get("adapter_model_sha256") != candidate.adapter_model_sha256
        or composite.get("adapter_config_sha256") != candidate.adapter_config_sha256
        or type(shape_audit) is not list
        or not shape_audit
        or any(
            type(item) is not dict or item.get("exact_rank_slice_audit") is not True
            for item in shape_audit
        )
    ):
        raise SkillAdapterCalibrationError("candidate composition proof differs")
    result = _read_json(candidate.training_result_path, _MAX_JSON_BYTES)
    source_result = _read_json(protocol.source_training_result.path, _MAX_JSON_BYTES)
    base_config_sha256 = next(
        item.sha256 for item in protocol.base_files if item.path.name == "config.json"
    )
    base_index_sha256 = next(
        item.sha256
        for item in protocol.base_files
        if item.path.name == "model.safetensors.index.json"
    )
    if (
        result.get("schema") != "jenny2.qlora-training.v1"
        or result.get("status") != "PASS"
        or result.get("base_revision") != BASE_REVISION
        or result.get("base_config_sha256") != base_config_sha256
        or result.get("base_index_sha256") != base_index_sha256
        or result.get("text_only_training") is not True
        or result.get("lora_scaling") != "standard-alpha-over-r"
        or result.get("adapter_artifact_kind")
        != "exact-rank-concatenated-skill-composite"
        or result.get("adapter_model_sha256") != candidate.adapter_model_sha256
        or result.get("adapter_config_sha256") != candidate.adapter_config_sha256
        or result.get("composition_manifest_ref") != candidate.manifest_ref
        or result.get("composition_manifest_sha256") != candidate.manifest_sha256
        or result.get("composition_source_training_result_sha256")
        != protocol.source_training_result.sha256
        or result.get("rank") != candidate.rank
        or result.get("training_profile") != source_result.get("training_profile")
        or result.get("training_sha256") != source_result.get("training_sha256")
        or result.get("evaluation_sha256") != source_result.get("evaluation_sha256")
        or result.get("curriculum_manifest_sha256")
        != source_result.get("curriculum_manifest_sha256")
        or tuple(result.get("target_modules", ())) != LORA_TARGET_MODULES
        or result.get("steps_completed") != source_result.get("steps_completed")
        or result.get("max_steps") != source_result.get("max_steps")
        or type(result.get("steps_completed")) is not int
        or result.get("steps_completed", 0) < 1
        or result.get("steps_completed") != result.get("max_steps")
        or type(result.get("held_out_teacher_forced_evaluation")) is not dict
    ):
        raise SkillAdapterCalibrationError("candidate packaged result differs")
    for field in ("training_sha256", "evaluation_sha256", "curriculum_manifest_sha256"):
        _require_sha(result.get(field), f"candidate result {field}")
    # Copied weight-1 metrics establish package lineage only. They remain
    # explicitly nonqualifying; selection uses fresh rows in this protocol.


def _verify_candidate_tensor_slices(
    candidate: CompiledCandidate, protocol: FrozenCalibrationProtocol
) -> None:
    """Independently prove the compiled tensor is the declared weighted sum."""

    # Delayed CPU-only imports keep module import and protocol inspection inert.
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:
        raise SkillAdapterCalibrationError(
            "independent tensor verifier dependencies are unavailable"
        ) from exc

    paths = (
        protocol.parent.path / "adapter_model.safetensors",
        protocol.residual.path / "adapter_model.safetensors",
        candidate.adapter_path / "adapter_model.safetensors",
    )
    try:
        with (
            safe_open(str(paths[0]), framework="numpy") as parent,
            safe_open(str(paths[1]), framework="numpy") as residual,
            safe_open(str(paths[2]), framework="numpy") as composite,
        ):
            if (
                parent.metadata() != {"format": "pt"}
                or residual.metadata() != {"format": "pt"}
                or composite.metadata() != {"format": "pt"}
            ):
                raise SkillAdapterCalibrationError(
                    "candidate tensor metadata differs"
                )
            parent_keys = set(parent.keys())
            if (
                not parent_keys
                or set(residual.keys()) != parent_keys
                or set(composite.keys()) != parent_keys
            ):
                raise SkillAdapterCalibrationError(
                    "candidate tensor key set differs from sources"
                )
            a_keys = {key for key in parent_keys if key.endswith(_LORA_A_SUFFIX)}
            b_keys = {key for key in parent_keys if key.endswith(_LORA_B_SUFFIX)}
            if parent_keys != a_keys | b_keys or not a_keys or len(a_keys) != len(b_keys):
                raise SkillAdapterCalibrationError("candidate LoRA tensor pairs differ")
            if {
                key[: -len(_LORA_A_SUFFIX)] + _LORA_B_SUFFIX for key in a_keys
            } != b_keys:
                raise SkillAdapterCalibrationError("candidate LoRA A/B keys differ")

            applied_weight = np.float32(float(Decimal(candidate.weight)))
            for a_key in sorted(a_keys):
                b_key = a_key[: -len(_LORA_A_SUFFIX)] + _LORA_B_SUFFIX
                parent_a = parent.get_tensor(a_key)
                parent_b = parent.get_tensor(b_key)
                residual_a = residual.get_tensor(a_key)
                residual_b = residual.get_tensor(b_key)
                composite_a = composite.get_tensor(a_key)
                composite_b = composite.get_tensor(b_key)
                tensors = (
                    parent_a,
                    parent_b,
                    residual_a,
                    residual_b,
                    composite_a,
                    composite_b,
                )
                if any(tensor.dtype != np.float32 for tensor in tensors) or any(
                    not bool(np.isfinite(tensor).all()) for tensor in tensors
                ):
                    raise SkillAdapterCalibrationError(
                        "candidate tensor dtype or finiteness differs"
                    )
                if (
                    parent_a.ndim != 2
                    or residual_a.shape != parent_a.shape
                    or parent_a.shape[0] != protocol.parent.rank
                    or parent_b.ndim != 2
                    or residual_b.shape != parent_b.shape
                    or parent_b.shape[1] != protocol.parent.rank
                    or composite_a.shape
                    != (candidate.rank, parent_a.shape[1])
                    or composite_b.shape
                    != (parent_b.shape[0], candidate.rank)
                ):
                    raise SkillAdapterCalibrationError(
                        "candidate tensor rank slices differ"
                    )
                split = protocol.parent.rank
                if (
                    not bool(np.array_equal(composite_a[:split], parent_a))
                    or not bool(
                        np.array_equal(
                            composite_a[split:], residual_a * applied_weight
                        )
                    )
                    or not bool(np.array_equal(composite_b[:, :split], parent_b))
                    or not bool(
                        np.array_equal(composite_b[:, split:], residual_b)
                    )
                ):
                    raise SkillAdapterCalibrationError(
                        "candidate tensor slices do not equal declared weighted sources"
                    )
    except SkillAdapterCalibrationError:
        raise
    except Exception as exc:
        raise SkillAdapterCalibrationError(
            "candidate tensor proof could not be read"
        ) from exc


def _evaluate_pair(
    backend: EvaluationBackend,
    target: EvaluationTarget,
    *,
    reading: Sequence[EvaluationRow],
    retention: Sequence[EvaluationRow],
    phase: str,
    maximum_tokens: int,
    pre_load_check: Callable[[], None],
) -> dict[str, object]:
    observed: dict[str, object] = {}
    for suite, rows in (("reading", reading), ("retention", retention)):
        pre_load_check()
        stats = backend.evaluate(
            target,
            rows,
            phase=phase,
            suite=suite,
            maximum_tokens=maximum_tokens,
        )
        pre_load_check()
        validated = _validate_statistics(rows, stats)
        observed[suite] = _summarize(validated)
    return observed


def _validate_statistics(
    rows: Sequence[EvaluationRow],
    statistics: Sequence[RowSufficientStatistic],
) -> tuple[RowSufficientStatistic, ...]:
    values = tuple(statistics)
    if len(values) != len(rows):
        raise SkillAdapterCalibrationError("evaluator row count differs")
    for row, item in zip(rows, values):
        if not isinstance(item, RowSufficientStatistic):
            raise SkillAdapterCalibrationError("evaluator statistic type differs")
        if (item.identity, item.skill, item.boundary) != (
            row.identity,
            row.skill,
            row.boundary,
        ):
            raise SkillAdapterCalibrationError("evaluator row identity differs")
        if (
            type(item.supervised_tokens) is not int
            or isinstance(item.supervised_tokens, bool)
            or item.supervised_tokens < 1
            or type(item.correct_tokens) is not int
            or isinstance(item.correct_tokens, bool)
            or not 0 <= item.correct_tokens <= item.supervised_tokens
            or type(item.nll_sum) not in (int, float)
            or isinstance(item.nll_sum, bool)
            or not math.isfinite(float(item.nll_sum))
            or float(item.nll_sum) < 0
        ):
            raise SkillAdapterCalibrationError("evaluator statistic is invalid")
    return values


def _summarize(statistics: Sequence[RowSufficientStatistic]) -> dict[str, object]:
    groups: dict[str, dict[str, int | float]] = {}
    rows_payload: list[dict[str, object]] = []
    for item in statistics:
        group = groups.setdefault(
            item.skill,
            {"examples": 0, "tokens": 0, "correct": 0, "nll_sum": 0.0},
        )
        group["examples"] += 1
        group["tokens"] += item.supervised_tokens
        group["correct"] += item.correct_tokens
        group["nll_sum"] += float(item.nll_sum)
        rows_payload.append(
            {
                "identity": item.identity,
                "skill": item.skill,
                "boundary": item.boundary,
                "supervised_tokens": item.supervised_tokens,
                "correct_tokens": item.correct_tokens,
                "nll_sum": float(item.nll_sum),
            }
        )
    by_skill: dict[str, dict[str, int | float]] = {}
    total_tokens = total_correct = total_examples = 0
    total_nll = 0.0
    for skill, group in sorted(groups.items()):
        tokens = int(group["tokens"])
        correct = int(group["correct"])
        nll_sum = float(group["nll_sum"])
        examples = int(group["examples"])
        by_skill[skill] = {
            "examples": examples,
            "tokens": tokens,
            "correct_tokens": correct,
            "nll_sum": nll_sum,
            "token_accuracy": correct / tokens,
            "mean_nll": nll_sum / tokens,
        }
        total_examples += examples
        total_tokens += tokens
        total_correct += correct
        total_nll += nll_sum
    if not by_skill or total_tokens < 1:
        raise SkillAdapterCalibrationError("evaluator summary is empty")
    return {
        "examples": total_examples,
        "tokens": total_tokens,
        "correct_tokens": total_correct,
        "nll_sum": total_nll,
        "micro_accuracy": total_correct / total_tokens,
        "micro_nll": total_nll / total_tokens,
        "macro_skill_accuracy": sum(
            float(item["token_accuracy"]) for item in by_skill.values()
        )
        / len(by_skill),
        "macro_skill_nll": sum(float(item["mean_nll"]) for item in by_skill.values())
        / len(by_skill),
        "by_skill": by_skill,
        "row_sufficient_statistics": rows_payload,
    }


def _parent_control_gates(retention: Mapping[str, object]) -> dict[str, bool]:
    by_skill = retention["by_skill"]
    assert isinstance(by_skill, dict)
    return {
        "complete_finite_metrics": _summary_finite(retention),
        "parent_retention_micro_accuracy": float(retention["micro_accuracy"])
        >= GATE_THRESHOLDS["parent_retention_micro_accuracy_min"]
        - _COMPARISON_EPSILON,
        "parent_retention_macro_accuracy": float(retention["macro_skill_accuracy"])
        >= GATE_THRESHOLDS["parent_retention_macro_skill_accuracy_min"]
        - _COMPARISON_EPSILON,
        "parent_retention_each_skill_accuracy": all(
            float(item["token_accuracy"])
            >= GATE_THRESHOLDS["parent_retention_each_skill_accuracy_min"]
            - _COMPARISON_EPSILON
            for item in by_skill.values()
        ),
        "parent_retention_micro_nll": float(retention["micro_nll"])
        <= GATE_THRESHOLDS["parent_retention_micro_nll_max"]
        + _COMPARISON_EPSILON,
        "parent_retention_macro_nll": float(retention["macro_skill_nll"])
        <= GATE_THRESHOLDS["parent_retention_macro_skill_nll_max"]
        + _COMPARISON_EPSILON,
    }


def _candidate_gates(
    parent: Mapping[str, object],
    candidate: Mapping[str, object],
    parent_gates: Mapping[str, bool],
) -> dict[str, bool]:
    reading_parent = parent["reading"]
    reading_candidate = candidate["reading"]
    retention_parent = parent["retention"]
    retention_candidate = candidate["retention"]
    assert isinstance(reading_parent, dict)
    assert isinstance(reading_candidate, dict)
    assert isinstance(retention_parent, dict)
    assert isinstance(retention_candidate, dict)
    rp_skill = reading_parent["by_skill"]
    rc_skill = reading_candidate["by_skill"]
    tp_skill = retention_parent["by_skill"]
    tc_skill = retention_candidate["by_skill"]
    assert isinstance(rp_skill, dict) and isinstance(rc_skill, dict)
    assert isinstance(tp_skill, dict) and isinstance(tc_skill, dict)
    same_skills = set(rp_skill) == set(rc_skill) and set(tp_skill) == set(tc_skill)
    gates = dict(parent_gates)
    gates.update(
        {
            "complete_finite_metrics": (
                gates.get("complete_finite_metrics", False)
                and _summary_finite(reading_parent)
                and _summary_finite(reading_candidate)
                and _summary_finite(retention_candidate)
                and same_skills
            ),
            "retention_micro_accuracy_noninferiority": (
                float(retention_parent["micro_accuracy"])
                - float(retention_candidate["micro_accuracy"])
                <= GATE_THRESHOLDS["retention_micro_accuracy_drop_max"]
                + _COMPARISON_EPSILON
            ),
            "retention_macro_accuracy_noninferiority": (
                float(retention_parent["macro_skill_accuracy"])
                - float(retention_candidate["macro_skill_accuracy"])
                <= GATE_THRESHOLDS["retention_macro_skill_accuracy_drop_max"]
                + _COMPARISON_EPSILON
            ),
            "retention_each_skill_accuracy_noninferiority": same_skills
            and all(
                float(tp_skill[skill]["token_accuracy"])
                - float(tc_skill[skill]["token_accuracy"])
                <= GATE_THRESHOLDS["retention_each_skill_accuracy_drop_max"]
                + _COMPARISON_EPSILON
                for skill in tp_skill
            ),
            "retention_micro_nll_noninferiority": (
                float(retention_candidate["micro_nll"])
                - float(retention_parent["micro_nll"])
                <= GATE_THRESHOLDS["retention_micro_nll_increase_max"]
                + _COMPARISON_EPSILON
            ),
            "retention_macro_nll_noninferiority": (
                float(retention_candidate["macro_skill_nll"])
                - float(retention_parent["macro_skill_nll"])
                <= GATE_THRESHOLDS["retention_macro_skill_nll_increase_max"]
                + _COMPARISON_EPSILON
            ),
            "retention_each_skill_nll_noninferiority": same_skills
            and all(
                float(tc_skill[skill]["mean_nll"])
                - float(tp_skill[skill]["mean_nll"])
                <= GATE_THRESHOLDS["retention_each_skill_nll_increase_max"]
                + _COMPARISON_EPSILON
                for skill in tp_skill
            ),
            "reading_micro_accuracy_superiority": (
                float(reading_candidate["micro_accuracy"])
                - float(reading_parent["micro_accuracy"])
                >= GATE_THRESHOLDS["reading_micro_accuracy_gain_min"]
                - _COMPARISON_EPSILON
            ),
            "reading_macro_accuracy_superiority": (
                float(reading_candidate["macro_skill_accuracy"])
                - float(reading_parent["macro_skill_accuracy"])
                >= GATE_THRESHOLDS["reading_macro_skill_accuracy_gain_min"]
                - _COMPARISON_EPSILON
            ),
            "reading_micro_nll_reduction": _ratio_gate(
                float(reading_candidate["micro_nll"]),
                float(reading_parent["micro_nll"]),
                GATE_THRESHOLDS["reading_micro_nll_ratio_max"],
            ),
            "reading_macro_nll_reduction": _ratio_gate(
                float(reading_candidate["macro_skill_nll"]),
                float(reading_parent["macro_skill_nll"]),
                GATE_THRESHOLDS["reading_macro_skill_nll_ratio_max"],
            ),
            "reading_each_skill_accuracy_floor": same_skills
            and all(
                float(rc_skill[skill]["token_accuracy"])
                - float(rp_skill[skill]["token_accuracy"])
                >= GATE_THRESHOLDS["reading_each_skill_accuracy_delta_min"]
                - _COMPARISON_EPSILON
                for skill in rp_skill
            ),
            "reading_each_skill_nll_floor": same_skills
            and all(
                float(rc_skill[skill]["mean_nll"])
                - float(rp_skill[skill]["mean_nll"])
                <= GATE_THRESHOLDS["reading_each_skill_nll_delta_max"]
                + _COMPARISON_EPSILON
                for skill in rp_skill
            ),
        }
    )
    return gates


def _ratio_gate(observed: float, baseline: float, maximum: float) -> bool:
    return (
        observed <= _COMPARISON_EPSILON
        if baseline == 0.0
        else observed <= maximum * baseline + _COMPARISON_EPSILON
    )


def _summary_finite(summary: Mapping[str, object]) -> bool:
    for key in ("micro_accuracy", "micro_nll", "macro_skill_accuracy", "macro_skill_nll"):
        value = summary.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            return False
    return True


def _result_payload(
    *,
    protocol: FrozenCalibrationProtocol,
    protocol_ref: str,
    pre_ref: str | None,
    status: str,
    calibration: Mapping[str, object] | None,
    selected: CompiledCandidate | None,
    final_confirmation: Mapping[str, object],
    error: Mapping[str, str] | None = None,
) -> dict[str, object]:
    confirmation = dict(final_confirmation)
    final_metrics = confirmation.get("metrics")
    if calibration is not None:
        confirmation["metrics"] = {
            "calibration": dict(calibration),
            "final": final_metrics if isinstance(final_metrics, dict) else {},
        }
    payload: dict[str, object] = {
        "schema": "jenny2.skill-adapter-calibration-result.v1",
        "status": status,
        "protocol_manifest_ref": protocol_ref,
        "pre_evaluation_manifest_ref": pre_ref,
        "source_training_result_sha256": protocol.source_training_result.sha256,
        "source_training_metrics_qualifying": False,
        "runtime_changed": False,
        "live_state_changed": False,
        "selected_candidate": (
            None if selected is None else _candidate_payload(selected)
        ),
        "final_confirmation": confirmation,
    }
    if error is not None:
        payload["error"] = dict(error)
    return payload


def _parent_target(component: ComponentIdentity) -> EvaluationTarget:
    return EvaluationTarget(
        "parent_control",
        None,
        component.path,
        component.adapter_model_sha256,
        component.adapter_config_sha256,
        component.rank,
        "source-parent-adapter",
    )


def _candidate_target(candidate: CompiledCandidate) -> EvaluationTarget:
    return EvaluationTarget(
        "weighted_composite",
        candidate.weight,
        candidate.adapter_path,
        candidate.adapter_model_sha256,
        candidate.adapter_config_sha256,
        candidate.rank,
        "exact-rank-concatenated-skill-composite",
    )


def _candidate_payload(candidate: CompiledCandidate) -> dict[str, object]:
    return {
        "weight": candidate.weight,
        "adapter_path": str(candidate.adapter_path),
        "adapter_model_sha256": candidate.adapter_model_sha256,
        "adapter_config_sha256": candidate.adapter_config_sha256,
        "composition_manifest_ref": candidate.manifest_ref,
        "composition_manifest_sha256": candidate.manifest_sha256,
        "training_result_sha256": candidate.training_result_sha256,
        "rank": candidate.rank,
    }


def _component_payload(
    component: ComponentIdentity, *, weight: str | None
) -> dict[str, object]:
    return {
        "name": component.name,
        "path": str(component.path),
        "adapter_model_sha256": component.adapter_model_sha256,
        "adapter_config_sha256": component.adapter_config_sha256,
        "rank": component.rank,
        "weight": weight,
    }


def _partition_payload(value: PartitionFingerprint) -> dict[str, object]:
    return {
        "count": value.count,
        "raw_jsonl_sha256": value.raw_jsonl_sha256,
        "identity_list_sha256": value.identity_list_sha256,
        "skill_counts": dict(sorted(value.skill_counts.items())),
    }


def _file_payload(value: FileIdentity) -> dict[str, str]:
    return {"path": str(value.path), "sha256": value.sha256}


def _raw_rows_sha(rows: Sequence[EvaluationRow]) -> str:
    return hashlib.sha256(b"".join(row.raw for row in rows)).hexdigest()


def _identity_list_sha(rows: Sequence[EvaluationRow]) -> str:
    return hashlib.sha256(
        "".join(row.identity + "\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _write_referenced_json(path: Path, payload: Mapping[str, object], *, ref_field: str) -> str:
    if ref_field in payload:
        raise SkillAdapterCalibrationError(f"payload already contains {ref_field}")
    ref = "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()
    _write_exclusive(path, _pretty_json({**payload, ref_field: ref}))
    return ref


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _new_output_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise SkillAdapterCalibrationError("output root must be canonical and absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parent = _real_directory(path.parent, "output parent")
    if path.parent.resolve() != parent:
        raise SkillAdapterCalibrationError("output parent differs")
    path.mkdir(mode=0o700)
    return path


def _validate_file(identity: FileIdentity, *, maximum_bytes: int) -> Path:
    _require_sha(identity.sha256, f"file {identity.path}")
    path = _real_file(identity.path, f"file {identity.path}", maximum_bytes)
    if _sha_file(path) != identity.sha256:
        raise SkillAdapterCalibrationError(f"SHA-256 differs for {path}")
    return path


def _real_directory(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise SkillAdapterCalibrationError(f"{label} must be a real absolute directory")
    resolved = path.resolve()
    if resolved != path:
        raise SkillAdapterCalibrationError(f"{label} path differs after resolution")
    return resolved


def _real_file(value: str | Path, label: str, maximum_bytes: int) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > maximum_bytes
    ):
        raise SkillAdapterCalibrationError(f"{label} must be a bounded real file")
    resolved = path.resolve()
    if resolved != path:
        raise SkillAdapterCalibrationError(f"{label} path differs after resolution")
    return resolved


def _read_json(path: Path, maximum_bytes: int) -> dict[str, object]:
    real = _real_file(path, f"JSON {path}", maximum_bytes)
    try:
        value = json.loads(real.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillAdapterCalibrationError(f"JSON is malformed at {path}") from exc
    if type(value) is not dict:
        raise SkillAdapterCalibrationError(f"JSON object required at {path}")
    return value


def _require_sha(value: object, label: str) -> None:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SkillAdapterCalibrationError(f"{label} SHA-256 is malformed")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _freeze_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        os.chmod(path, 0o444)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
