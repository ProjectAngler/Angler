"""Runtime-attested Jenny 2.0 composition manifests.

This module is a pure construction boundary used by the live runtime wiring.
It accepts one explicit, already-validated snapshot of the assembled runtime and emits the
``jenny2.live-composition-manifest.v1`` record described by
``JENNY_2_LIVE_COMPOSITION_MANIFEST_V1.schema.json``.  It does not discover a
runtime, acquire the shared operation lock, change live state, grant a
permission, select an affordance for the model, or make a qualification/
promotion claim.  Its caller is responsible for taking every supplied object
under the shared operation lock; the runtime and API integrations do so.

The aggregate integrity state is derived rather than caller-selected.  A
manifest can be ``FRESH`` only when every required verification is explicitly
present, no component was marked stale, and all cross-field/file checks agree.
Raw exception text is represented only by a digest, and the compact cognitive
projection contains no filesystem paths.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Literal
import unicodedata

from .jenny_genesis import GENESIS_CONTRACT, OWNER_AGENT_REF, JennyGenesis
from .persistent_autonomy import (
    SUPERVISOR_STATE_CONTRACT,
    Affordance,
    AutonomyHeartbeat,
    PermissionGate,
    SupervisorStateHead,
)
from .temporal_v2 import TEMPORAL_NOW_CONTRACT, TemporalNow


MANIFEST_SCHEMA = "jenny2.live-composition-manifest.v1"
MANIFEST_VERSION = "1.0.0"
MANIFEST_SCHEMA_ID = "urn:angler:jenny2:live-composition-manifest:v1"
MANIFEST_HASH_RULE = (
    "sha256 of UTF-8 canonical JSON using sorted keys, no insignificant "
    "whitespace, no NaN/Infinity, with manifest_ref omitted"
)
GUIDE_DOCUMENT_ID = "ANG-REPORT-JENNY-2-SYSTEM-GUIDE-001"
GUIDE_CONTRACT = "jenny2.system-guide.v1"
GUIDE_VERSION = "1.0.0"
TOOL_CONTEXT_POLICY = (
    "compact cards in ordinary context; exact full definition loaded only "
    "when selected"
)

SCHEDULER_CONFIGURATION_SCHEMA = "jenny2.scheduler-configuration.v1"
AFFORDANCE_CARD_SCHEMA = "jenny2.affordance-card.v1"
TOOL_REGISTRY_SCHEMA = "jenny2.affordance-registry.v1"
EFFECT_POLICY_SCHEMA = "jenny2.effect-policy.v1"

REQUIRED_VERIFICATIONS = frozenset(
    {
        "agent",
        "guide",
        "schema_artifact",
        "model_binding",
        "canonical_state",
        "time",
        "snapshot_coherence",
        "scheduler",
        "memory",
        "tools",
        "effects",
        "evidence_refs",
    }
)
_OPERATIONAL_VERIFICATIONS = frozenset(
    {
        "model_binding",
        "canonical_state",
        "time",
        "snapshot_coherence",
        "scheduler",
        "memory",
        "tools",
        "effects",
    }
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_AFFORDANCE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*$")
_SCOPE = _AFFORDANCE_ID
_SERVED_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,511}$")
_MAX_LOCAL_ARTIFACT_BYTES = 16 * 1024 * 1024
_MANIFEST_BUILD_TOKEN = object()


IntegrityStatus = Literal["FRESH", "STALE", "MISMATCH", "INCOMPLETE"]
OperationalStatus = Literal["READY", "DEGRADED", "STOPPED", "UNKNOWN"]
VerificationDisposition = Literal["EXACT", "STALE", "MISMATCH", "UNAVAILABLE"]


class CompositionInputError(ValueError):
    """The supplied snapshot cannot be represented by the v1 schema."""


class CompositionArtifactError(CompositionInputError):
    """A local artifact could not be safely and exactly hashed."""


class CompositionIntegrityError(RuntimeError):
    """A supposedly content-addressed emitted manifest failed revalidation."""


def canonical_json(value: object) -> str:
    """Return this module's deterministic UTF-8 JSON identity form.

    The v1 report requires sorted keys, no insignificant whitespace, and no
    non-finite numbers.  NFC is required here (rather than silently repaired)
    so two visually equivalent strings cannot acquire accidental identities.
    """

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # defensive after validation
        raise CompositionInputError("value is not canonical JSON data") from exc


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8", errors="strict")


def content_ref(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_json_value(value: object, *, _depth: int = 0) -> None:
    if _depth > 64:
        raise CompositionInputError("canonical JSON exceeds the nesting ceiling")
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CompositionInputError("canonical JSON forbids NaN and Infinity")
        return
    if type(value) is str:
        if unicodedata.normalize("NFC", value) != value:
            raise CompositionInputError("canonical JSON strings must use NFC Unicode")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CompositionInputError("canonical JSON must be valid UTF-8") from exc
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item, _depth=_depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CompositionInputError("canonical JSON object keys must be text")
            _validate_json_value(key, _depth=_depth + 1)
            _validate_json_value(item, _depth=_depth + 1)
        return
    raise CompositionInputError("value contains a non-JSON type")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise CompositionInputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ref(value: object, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or _REFERENCE.fullmatch(value) is None:
        raise CompositionInputError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise CompositionInputError(f"{label} must be bounded non-empty text")
    if "\x00" in value or any(character in "\r\n" for character in value):
        raise CompositionInputError(f"{label} contains an unsafe control character")
    if unicodedata.normalize("NFC", value) != value:
        raise CompositionInputError(f"{label} must use NFC Unicode")
    return value


def _timestamp(value: object, label: str) -> str:
    if type(value) is not str:
        raise CompositionInputError(f"{label} must be an RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CompositionInputError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.utcoffset() is None:
        raise CompositionInputError(f"{label} must include a UTC offset")
    return value


def _sorted_unique(
    values: Sequence[str],
    label: str,
    *,
    maximum: int,
    validator: object | None = None,
) -> list[str]:
    if type(values) not in (tuple, list):
        raise CompositionInputError(f"{label} must be a finite sequence")
    if len(values) > maximum:
        raise CompositionInputError(f"{label} exceeds its item ceiling")
    result: list[str] = []
    for item in values:
        if type(item) is not str:
            raise CompositionInputError(f"{label} items must be text")
        if validator is not None and callable(validator):
            validator(item, f"{label} item")
        else:
            _text(item, f"{label} item", 256)
        result.append(item)
    if len(set(result)) != len(result):
        raise CompositionInputError(f"{label} must not contain duplicates")
    return sorted(result)


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompositionArtifactError("local artifact could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompositionArtifactError("local artifact is not a regular file")
        if before.st_size > maximum_bytes:
            raise CompositionArtifactError("local artifact exceeds its byte ceiling")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1_048_576, maximum_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise CompositionArtifactError("local artifact exceeds its byte ceiling")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or total != after.st_size:
            raise CompositionArtifactError("local artifact changed while being hashed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _raw_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalFileArtifactSnapshot:
    """A local file and the non-secret path recorded in the manifest.

    ``source_path`` is used only to hash bytes and is never placed in the
    cognitive projection.  ``manifest_path`` should normally be a repository-
    relative or otherwise owner-approved public locator.  An expected digest
    is optional; when supplied, disagreement produces ``MISMATCH`` rather than
    silently rebinding the old attestation.
    """

    source_path: Path
    manifest_path: str
    expected_sha256: str | None = None
    maximum_bytes: int = _MAX_LOCAL_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            object.__setattr__(self, "source_path", Path(self.source_path))
        _text(self.manifest_path, "artifact manifest_path", 4096)
        public_path = PurePosixPath(self.manifest_path)
        if (
            public_path.is_absolute()
            or public_path.as_posix() != self.manifest_path
            or any(part in ("", ".", "..") for part in public_path.parts)
        ):
            raise CompositionInputError(
                "artifact manifest_path must be a canonical relative public locator"
            )
        if self.expected_sha256 is not None:
            _digest(self.expected_sha256, "artifact expected_sha256")
        if type(self.maximum_bytes) is not int or not 1 <= self.maximum_bytes <= (
            256 * 1024 * 1024
        ):
            raise CompositionInputError("artifact maximum_bytes is out of bounds")

    @classmethod
    def capture(
        cls,
        source_path: str | Path,
        *,
        manifest_path: str,
        maximum_bytes: int = _MAX_LOCAL_ARTIFACT_BYTES,
    ) -> "LocalFileArtifactSnapshot":
        source = Path(source_path)
        observed = _raw_sha256(
            _read_regular_file(source, maximum_bytes=maximum_bytes)
        )
        return cls(
            source_path=source,
            manifest_path=manifest_path,
            expected_sha256=observed,
            maximum_bytes=maximum_bytes,
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactSnapshot:
    path: str
    revision: str
    config_sha256: str
    index_sha256: str

    def __post_init__(self) -> None:
        _text(self.path, "model artifact path", 4096)
        _text(self.revision, "model artifact revision", 256)
        _digest(self.config_sha256, "model config_sha256")
        _digest(self.index_sha256, "model index_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
        }


@dataclass(frozen=True, slots=True)
class AdapterSnapshot:
    path: str
    model_sha256: str
    config_sha256: str
    rank: int
    target_modules: tuple[str, ...]
    training_result_sha256: str
    curriculum_manifest_sha256: str

    def __post_init__(self) -> None:
        _text(self.path, "adapter path", 4096)
        _digest(self.model_sha256, "adapter model_sha256")
        _digest(self.config_sha256, "adapter config_sha256")
        if type(self.rank) is not int or not 1 <= self.rank <= 64:
            raise CompositionInputError("adapter rank must be 1 through 64")
        _sorted_unique(
            self.target_modules,
            "adapter target_modules",
            maximum=64,
        )
        if not self.target_modules:
            raise CompositionInputError("adapter target_modules cannot be empty")
        _digest(self.training_result_sha256, "adapter training_result_sha256")
        _digest(
            self.curriculum_manifest_sha256,
            "adapter curriculum_manifest_sha256",
        )

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
            "rank": self.rank,
            "target_modules": _sorted_unique(
                self.target_modules,
                "adapter target_modules",
                maximum=64,
            ),
            "training_result_sha256": self.training_result_sha256,
            "curriculum_manifest_sha256": self.curriculum_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelBindingSnapshot:
    kind: Literal["BASE_CONTROL", "LORA"]
    binding_ref: str | None
    binding_file: LocalFileArtifactSnapshot | None
    base_served_model: str
    configured_served_model: str
    observed_served_model: str
    runtime_ref: str
    runtime_image: str
    runtime_revision: str
    source_model: ModelArtifactSnapshot
    quantized_model: ModelArtifactSnapshot
    adapter: AdapterSnapshot | None
    selector_qualification_ref: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("BASE_CONTROL", "LORA"):
            raise CompositionInputError("model binding kind is unsupported")
        _ref(self.binding_ref, "model binding_ref", nullable=True)
        for label, value in (
            ("base_served_model", self.base_served_model),
            ("configured_served_model", self.configured_served_model),
            ("observed_served_model", self.observed_served_model),
        ):
            if type(value) is not str or _SERVED_MODEL.fullmatch(value) is None:
                raise CompositionInputError(f"model binding {label} is malformed")
        _ref(self.runtime_ref, "model runtime_ref")
        _text(self.runtime_image, "model runtime_image", 1024)
        _text(self.runtime_revision, "model runtime_revision", 256)
        if not isinstance(self.source_model, ModelArtifactSnapshot):
            raise CompositionInputError("source_model must be a model snapshot")
        if not isinstance(self.quantized_model, ModelArtifactSnapshot):
            raise CompositionInputError("quantized_model must be a model snapshot")
        _ref(
            self.selector_qualification_ref,
            "selector_qualification_ref",
            nullable=True,
        )
        if self.kind == "LORA":
            if self.binding_ref is None or self.binding_file is None:
                raise CompositionInputError("LORA requires a binding ref and file")
            if not isinstance(self.adapter, AdapterSnapshot):
                raise CompositionInputError("LORA requires an adapter snapshot")
        elif any(
            item is not None
            for item in (self.binding_ref, self.binding_file, self.adapter)
        ):
            raise CompositionInputError(
                "BASE_CONTROL cannot carry a binding file, ref, or adapter"
            )

    @classmethod
    def from_lora_binding(
        cls,
        binding: object,
        *,
        binding_file: LocalFileArtifactSnapshot,
        observed_served_model: str,
    ) -> "ModelBindingSnapshot":
        """Project a validated ``SGLangLoRABinding`` without its endpoint."""

        required = (
            "binding_ref",
            "base_served_model",
            "served_model",
            "runtime_ref",
            "runtime_image",
            "runtime_revision",
            "source_model_path",
            "source_revision",
            "source_config_sha256",
            "source_index_sha256",
            "quantized_model_path",
            "quantized_revision",
            "quantized_config_sha256",
            "quantized_index_sha256",
            "adapter_path",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "rank",
            "target_modules",
            "training_result_sha256",
            "curriculum_manifest_sha256",
            "selector_qualification_ref",
        )
        if any(not hasattr(binding, name) for name in required):
            raise CompositionInputError("validated LoRA binding fields are incomplete")
        return cls(
            kind="LORA",
            binding_ref=binding.binding_ref,
            binding_file=binding_file,
            base_served_model=binding.base_served_model,
            configured_served_model=binding.served_model,
            observed_served_model=observed_served_model,
            runtime_ref=binding.runtime_ref,
            runtime_image=binding.runtime_image,
            runtime_revision=binding.runtime_revision,
            source_model=ModelArtifactSnapshot(
                path=binding.source_model_path,
                revision=binding.source_revision,
                config_sha256=binding.source_config_sha256,
                index_sha256=binding.source_index_sha256,
            ),
            quantized_model=ModelArtifactSnapshot(
                path=binding.quantized_model_path,
                revision=binding.quantized_revision,
                config_sha256=binding.quantized_config_sha256,
                index_sha256=binding.quantized_index_sha256,
            ),
            adapter=AdapterSnapshot(
                path=binding.adapter_path,
                model_sha256=binding.adapter_model_sha256,
                config_sha256=binding.adapter_config_sha256,
                rank=binding.rank,
                target_modules=tuple(binding.target_modules),
                training_result_sha256=binding.training_result_sha256,
                curriculum_manifest_sha256=binding.curriculum_manifest_sha256,
            ),
            selector_qualification_ref=binding.selector_qualification_ref,
        )


@dataclass(frozen=True, slots=True)
class CanonicalStateSnapshot:
    head: SupervisorStateHead
    store_identity_ref: str
    pending_ref: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.head, SupervisorStateHead):
            raise CompositionInputError("state head must be SupervisorStateHead")
        _ref(self.store_identity_ref, "state store_identity_ref")
        _ref(self.pending_ref, "state pending_ref", nullable=True)
        _ref(self.head.state_ref, "state_ref")
        _ref(self.head.last_event_ref, "state last_event_ref", nullable=True)
        if type(self.head.revision) is not int or self.head.revision < 0:
            raise CompositionInputError("state revision must be non-negative")
        if (
            type(self.head.moving_origin_ordinal) is not int
            or self.head.moving_origin_ordinal < -1
        ):
            raise CompositionInputError("state Moving Origin ordinal is invalid")
        if type(self.head.scheduler_enabled) is not bool:
            raise CompositionInputError("state scheduler flag must be boolean")


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    implementation_ref: str
    enabled: bool
    life_loop_running: bool
    interval_seconds: float
    max_steps_per_session: int
    error: object | None = None
    expected_configuration_ref: str | None = None

    def __post_init__(self) -> None:
        _ref(self.implementation_ref, "scheduler implementation_ref")
        if type(self.enabled) is not bool or type(self.life_loop_running) is not bool:
            raise CompositionInputError("scheduler status fields must be boolean")
        if (
            type(self.interval_seconds) not in (int, float)
            or not math.isfinite(float(self.interval_seconds))
            or not 0.001 <= float(self.interval_seconds) <= 3600
        ):
            raise CompositionInputError("scheduler interval_seconds is out of bounds")
        if (
            type(self.max_steps_per_session) is not int
            or not 1 <= self.max_steps_per_session <= 10_000
        ):
            raise CompositionInputError(
                "scheduler max_steps_per_session is out of bounds"
            )
        _ref(
            self.expected_configuration_ref,
            "scheduler expected_configuration_ref",
            nullable=True,
        )

    @classmethod
    def from_heartbeat(
        cls,
        heartbeat: AutonomyHeartbeat,
        *,
        implementation_ref: str,
        enabled: bool,
        expected_configuration_ref: str | None = None,
    ) -> "SchedulerSnapshot":
        if not isinstance(heartbeat, AutonomyHeartbeat):
            raise CompositionInputError("heartbeat must be AutonomyHeartbeat")
        return cls(
            implementation_ref=implementation_ref,
            enabled=enabled,
            life_loop_running=heartbeat.is_running,
            interval_seconds=heartbeat.interval_seconds,
            max_steps_per_session=heartbeat.max_steps_per_session,
            error=heartbeat.error,
            expected_configuration_ref=expected_configuration_ref,
        )

    def configuration_payload(self) -> dict[str, object]:
        return {
            "schema": SCHEDULER_CONFIGURATION_SCHEMA,
            "enabled": self.enabled,
            "interval_seconds": float(self.interval_seconds),
            "max_steps_per_session": self.max_steps_per_session,
        }

    @property
    def configuration_ref(self) -> str:
        return content_ref(self.configuration_payload())


@dataclass(frozen=True, slots=True)
class CanonicalMemorySnapshot:
    canonical_writer_ref: str
    canonical_store_ref: str
    projection_backend_refs: tuple[str, ...]
    pending_projections: int
    projection_error: object | None = None

    def __post_init__(self) -> None:
        _ref(self.canonical_writer_ref, "memory canonical_writer_ref")
        _ref(self.canonical_store_ref, "memory canonical_store_ref")
        _sorted_unique(
            self.projection_backend_refs,
            "memory projection_backend_refs",
            maximum=32,
            validator=_ref,
        )
        if type(self.pending_projections) is not int or self.pending_projections < 0:
            raise CompositionInputError(
                "memory pending_projections must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class AffordanceSnapshot:
    affordance: Affordance
    full_definition_ref: str
    executor_ref: str | None
    observable_source_ref: str | None
    execution_mode: Literal["SYNC", "DEFERRED"] = "SYNC"
    availability: Literal["ACTIVE", "INACTIVE", "DEGRADED"] = "ACTIVE"
    expected_card_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.affordance, Affordance):
            raise CompositionInputError("affordance must be an Affordance")
        _ref(self.full_definition_ref, "affordance full_definition_ref")
        _ref(self.executor_ref, "affordance executor_ref", nullable=True)
        _ref(
            self.observable_source_ref,
            "affordance observable_source_ref",
            nullable=True,
        )
        _ref(self.expected_card_ref, "affordance expected_card_ref", nullable=True)
        if self.execution_mode not in ("SYNC", "DEFERRED"):
            raise CompositionInputError("affordance execution_mode is unsupported")
        if self.availability not in ("ACTIVE", "INACTIVE", "DEGRADED"):
            raise CompositionInputError("affordance availability is unsupported")
        if (
            self.affordance.authorization_mode == "HOST_GUARDED"
            and self.execution_mode != "DEFERRED"
        ):
            raise CompositionInputError(
                "host-guarded affordances require deferred execution"
            )

    def card_payload(self) -> dict[str, object]:
        return {
            "schema": AFFORDANCE_CARD_SCHEMA,
            "affordance_id": self.affordance.affordance_id,
            "disposition": self.affordance.disposition,
            "description": self.affordance.description,
            "permission_scope": self.affordance.permission_scope,
            "external_effect": self.affordance.external_effect,
            "authorization_mode": self.affordance.authorization_mode,
        }

    @property
    def card_ref(self) -> str:
        return content_ref(self.card_payload())

    def entry_payload(self) -> dict[str, object]:
        return {
            "affordance_id": self.affordance.affordance_id,
            "card_ref": self.card_ref,
            "full_definition_ref": self.full_definition_ref,
            "executor_ref": self.executor_ref,
            "observable_source_ref": self.observable_source_ref,
            "permission_scope": self.affordance.permission_scope,
            "external_effect": self.affordance.external_effect,
            "authorization_mode": self.affordance.authorization_mode,
            "execution_mode": self.execution_mode,
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class ToolRegistrySnapshot:
    entries: tuple[AffordanceSnapshot, ...]
    expected_registry_ref: str | None = None

    def __post_init__(self) -> None:
        if type(self.entries) not in (tuple, list):
            raise CompositionInputError("tool entries must be a finite sequence")
        if type(self.entries) is list:
            object.__setattr__(self, "entries", tuple(self.entries))
        if len(self.entries) > 512:
            raise CompositionInputError("tool registry exceeds 512 entries")
        if any(not isinstance(item, AffordanceSnapshot) for item in self.entries):
            raise CompositionInputError("tool registry entries must be snapshots")
        identifiers = [item.affordance.affordance_id for item in self.entries]
        if len(set(identifiers)) != len(identifiers):
            raise CompositionInputError("tool registry repeats an affordance_id")
        _ref(
            self.expected_registry_ref,
            "tools expected_registry_ref",
            nullable=True,
        )

    def canonical_entries(self) -> list[dict[str, object]]:
        return [
            item.entry_payload()
            for item in sorted(
                self.entries, key=lambda entry: entry.affordance.affordance_id
            )
        ]

    @property
    def registry_ref(self) -> str:
        return content_ref(
            {
                "schema": TOOL_REGISTRY_SCHEMA,
                "context_policy": TOOL_CONTEXT_POLICY,
                "entries": self.canonical_entries(),
            }
        )


@dataclass(frozen=True, slots=True)
class EffectPolicySnapshot:
    general_external_effects_enabled: bool
    allowed_internal_scopes: tuple[str, ...]
    allowed_read_only_scopes: tuple[str, ...]
    allowed_host_guarded_scopes: tuple[str, ...]
    host_guarded_tools_enabled: bool
    host_catalog_ref: str | None
    expected_policy_ref: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.general_external_effects_enabled) is not bool
            or type(self.host_guarded_tools_enabled) is not bool
        ):
            raise CompositionInputError("effect policy flags must be boolean")
        for name in (
            "allowed_internal_scopes",
            "allowed_read_only_scopes",
            "allowed_host_guarded_scopes",
        ):
            value = getattr(self, name)
            if type(value) is list:
                object.__setattr__(self, name, tuple(value))
                value = getattr(self, name)
            _sorted_unique(value, f"effects {name}", maximum=64, validator=_scope)
        _ref(self.host_catalog_ref, "effects host_catalog_ref", nullable=True)
        _ref(
            self.expected_policy_ref,
            "effects expected_policy_ref",
            nullable=True,
        )

    @classmethod
    def from_permission_gate(
        cls,
        permission_gate: PermissionGate,
        *,
        host_guarded_tools_enabled: bool,
        host_catalog_ref: str | None,
        expected_policy_ref: str | None = None,
    ) -> "EffectPolicySnapshot":
        if not isinstance(permission_gate, PermissionGate):
            raise CompositionInputError("permission_gate must be PermissionGate")
        return cls(
            general_external_effects_enabled=(
                permission_gate.external_effects_enabled
            ),
            allowed_internal_scopes=tuple(permission_gate.allowed_internal_scopes),
            allowed_read_only_scopes=tuple(
                permission_gate.allowed_external_read_scopes
            ),
            allowed_host_guarded_scopes=tuple(
                permission_gate.allowed_host_guarded_scopes
            ),
            host_guarded_tools_enabled=host_guarded_tools_enabled,
            host_catalog_ref=host_catalog_ref,
            expected_policy_ref=expected_policy_ref,
        )

    def policy_payload(self) -> dict[str, object]:
        return {
            "schema": EFFECT_POLICY_SCHEMA,
            "general_external_effects_enabled": (
                self.general_external_effects_enabled
            ),
            # Internal scopes are committed by policy_ref even though the v1
            # schema exposes them only on individual affordance entries.
            "allowed_internal_scopes": _sorted_unique(
                self.allowed_internal_scopes,
                "effects allowed_internal_scopes",
                maximum=64,
                validator=_scope,
            ),
            "allowed_read_only_scopes": _sorted_unique(
                self.allowed_read_only_scopes,
                "effects allowed_read_only_scopes",
                maximum=64,
                validator=_scope,
            ),
            "allowed_host_guarded_scopes": _sorted_unique(
                self.allowed_host_guarded_scopes,
                "effects allowed_host_guarded_scopes",
                maximum=64,
                validator=_scope,
            ),
            "host_guarded_tools_enabled": self.host_guarded_tools_enabled,
            "host_catalog_ref": self.host_catalog_ref,
        }

    @property
    def policy_ref(self) -> str:
        return content_ref(self.policy_payload())

    def manifest_payload(self) -> dict[str, object]:
        return {
            "policy_ref": self.policy_ref,
            "general_external_effects_enabled": (
                self.general_external_effects_enabled
            ),
            "allowed_read_only_scopes": _sorted_unique(
                self.allowed_read_only_scopes,
                "effects allowed_read_only_scopes",
                maximum=64,
                validator=_scope,
            ),
            "allowed_host_guarded_scopes": _sorted_unique(
                self.allowed_host_guarded_scopes,
                "effects allowed_host_guarded_scopes",
                maximum=64,
                validator=_scope,
            ),
            "host_guarded_tools_enabled": self.host_guarded_tools_enabled,
            "host_catalog_ref": self.host_catalog_ref,
        }


def _scope(value: object, label: str) -> str:
    if type(value) is not str or _SCOPE.fullmatch(value) is None:
        raise CompositionInputError(f"{label} must be a canonical permission scope")
    return value


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Evidence-bound disposition for one exact captured component.

    The receipt is intentionally supplied by an external validator.  The
    builder independently derives ``subject_ref`` from the captured bytes and
    fields, then requires ``evidence_ref`` to be present in the manifest's
    evidence ledger.  This prevents a bare list of component names from
    manufacturing ``FRESH``.
    """

    component: str
    disposition: VerificationDisposition
    subject_ref: str | None
    evidence_ref: str | None

    def __post_init__(self) -> None:
        if self.component not in REQUIRED_VERIFICATIONS:
            raise CompositionInputError("verification receipt component is unknown")
        if self.disposition not in (
            "EXACT",
            "STALE",
            "MISMATCH",
            "UNAVAILABLE",
        ):
            raise CompositionInputError(
                "verification receipt disposition is unsupported"
            )
        if self.disposition == "UNAVAILABLE":
            if self.subject_ref is not None or self.evidence_ref is not None:
                raise CompositionInputError(
                    "an unavailable verification cannot claim subject or evidence"
                )
            return
        _ref(self.subject_ref, "verification receipt subject_ref")
        _ref(self.evidence_ref, "verification receipt evidence_ref")


@dataclass(frozen=True, slots=True)
class CompositionVerification:
    """External verification receipts for the caller-locked snapshot."""

    receipts: tuple[VerificationReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.receipts) is list:
            object.__setattr__(self, "receipts", tuple(self.receipts))
        if type(self.receipts) is not tuple or any(
            not isinstance(item, VerificationReceipt) for item in self.receipts
        ):
            raise CompositionInputError(
                "verification receipts must be a finite tuple of receipts"
            )
        components = [item.component for item in self.receipts]
        if len(set(components)) != len(components):
            raise CompositionInputError(
                "verification receipts repeat a component"
            )
        object.__setattr__(
            self,
            "receipts",
            tuple(sorted(self.receipts, key=lambda item: item.component)),
        )

    def by_component(self) -> dict[str, VerificationReceipt]:
        return {item.component: item for item in self.receipts}


@dataclass(frozen=True, slots=True)
class JennyCompositionSnapshot:
    genesis: JennyGenesis
    guide_artifact: LocalFileArtifactSnapshot
    schema_artifact: LocalFileArtifactSnapshot
    model_binding: ModelBindingSnapshot
    canonical_state: CanonicalStateSnapshot
    time: TemporalNow
    scheduler: SchedulerSnapshot
    memory: CanonicalMemorySnapshot
    tools: ToolRegistrySnapshot
    effects: EffectPolicySnapshot
    evidence_refs: tuple[str, ...]
    verification: CompositionVerification

    def __post_init__(self) -> None:
        exact_types = (
            (self.genesis, JennyGenesis, "genesis"),
            (self.guide_artifact, LocalFileArtifactSnapshot, "guide_artifact"),
            (self.schema_artifact, LocalFileArtifactSnapshot, "schema_artifact"),
            (self.model_binding, ModelBindingSnapshot, "model_binding"),
            (self.canonical_state, CanonicalStateSnapshot, "canonical_state"),
            (self.time, TemporalNow, "time"),
            (self.scheduler, SchedulerSnapshot, "scheduler"),
            (self.memory, CanonicalMemorySnapshot, "memory"),
            (self.tools, ToolRegistrySnapshot, "tools"),
            (self.effects, EffectPolicySnapshot, "effects"),
            (self.verification, CompositionVerification, "verification"),
        )
        for value, expected, label in exact_types:
            if not isinstance(value, expected):
                raise CompositionInputError(f"{label} has the wrong snapshot type")
        if type(self.evidence_refs) is list:
            object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        _sorted_unique(
            self.evidence_refs,
            "evidence_refs",
            maximum=256,
            validator=_ref,
        )


class JennyCompositionManifest(Mapping[str, object]):
    """Immutable JSON-backed manifest plus its derived cognitive projection."""

    __slots__ = ("_manifest_json", "_projection_json")

    def __init__(
        self,
        manifest: Mapping[str, object],
        *,
        _build_token: object | None = None,
    ) -> None:
        if _build_token is not _MANIFEST_BUILD_TOKEN:
            raise CompositionIntegrityError(
                "composition manifests must be created by the validated builder"
            )
        self._manifest_json = canonical_json(dict(manifest))
        self._projection_json = canonical_json(_project_validated_payload(dict(manifest)))

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @property
    def manifest_ref(self) -> str:
        value = self.to_dict()["manifest_ref"]
        assert type(value) is str
        return value

    @property
    def integrity_status(self) -> IntegrityStatus:
        value = self.to_dict()["integrity_status"]
        assert value in ("FRESH", "STALE", "MISMATCH", "INCOMPLETE")
        return value  # type: ignore[return-value]

    @property
    def payload(self) -> dict[str, object]:
        return self.to_dict()

    @property
    def self_world(self) -> dict[str, object]:
        return json.loads(self._projection_json)

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._manifest_json)

    def canonical_json(self) -> str:
        return self._manifest_json

    def canonical_bytes(self) -> bytes:
        return self._manifest_json.encode("utf-8")

    def self_world_projection(self) -> dict[str, object]:
        return self.self_world


class JennyCompositionManifestBuilder:
    """Build one manifest from a caller-locked runtime snapshot."""

    def build(self, snapshot: JennyCompositionSnapshot) -> JennyCompositionManifest:
        if not isinstance(snapshot, JennyCompositionSnapshot):
            raise CompositionInputError("snapshot must be JennyCompositionSnapshot")

        mismatches: set[str] = set()

        schema_payload, schema_bytes = _artifact_payload(
            snapshot.schema_artifact,
            "schema_artifact",
            mismatches,
        )
        guide_payload, guide_bytes = _artifact_payload(
            snapshot.guide_artifact,
            "guide",
            mismatches,
        )
        _verify_schema_contract(schema_bytes, mismatches)
        _verify_guide_contract(guide_bytes, mismatches)

        genesis = snapshot.genesis
        if genesis.agent_ref != OWNER_AGENT_REF or genesis.contract != GENESIS_CONTRACT:
            raise CompositionInputError("snapshot is not the Jenny 2.0 genesis")
        genesis_ref = _ref(genesis.genesis_ref, "genesis_ref")
        recomputed_genesis_ref = "sha256:" + hashlib.sha256(
            genesis.canonical_bytes()
        ).hexdigest()
        if genesis_ref != recomputed_genesis_ref:
            mismatches.add("agent.genesis_ref: differs from canonical genesis bytes")
        owner_genesis = JennyGenesis.owner_approved(
            created_at_utc=genesis.created_at_utc
        )
        if genesis.canonical_bytes() != owner_genesis.canonical_bytes():
            mismatches.add("agent.genesis_ref: differs from the owner-pinned genesis")

        binding = snapshot.model_binding
        binding_file_payload: dict[str, object] | None = None
        binding_file_bytes: bytes | None = None
        if binding.binding_file is not None:
            binding_file_payload, binding_file_bytes = _artifact_payload(
                binding.binding_file,
                "model_binding.binding_file",
                mismatches,
            )
        if binding.configured_served_model != binding.observed_served_model:
            mismatches.add(
                "model_binding.configured_served_model: differs from the observed served model"
            )
        if binding.kind == "LORA":
            assert binding.adapter is not None
            expected_served_model = (
                binding.base_served_model
                + "--lora-sha256-"
                + binding.adapter.model_sha256
            )
            if binding.configured_served_model != expected_served_model:
                mismatches.add(
                    "model_binding.adapter.model_sha256: differs from the configured served model"
                )
            assert binding_file_bytes is not None
            _verify_lora_binding_file(binding_file_bytes, binding, mismatches)

        state = snapshot.canonical_state
        temporal = snapshot.time
        if temporal.contract != TEMPORAL_NOW_CONTRACT:
            raise CompositionInputError("trusted-time contract is unsupported")
        _timestamp(temporal.trusted_utc, "time trusted_utc")
        _text(temporal.local_time, "time local_time", 128)
        _text(temporal.local_timezone, "time local_timezone", 128)
        _ref(temporal.sample_ref, "time sample_ref")
        _ref(temporal.clock_anchor_ref, "time clock_anchor_ref")
        if temporal.moving_origin_ordinal != state.head.moving_origin_ordinal:
            mismatches.add(
                "canonical_state.moving_origin_ordinal: differs from the trusted-time sample"
            )
        if state.head.scheduler_enabled != snapshot.scheduler.enabled:
            mismatches.add(
                "scheduler.enabled: differs from the canonical state head"
            )
        if state.store_identity_ref != snapshot.memory.canonical_store_ref:
            mismatches.add(
                "memory.canonical_store_ref: differs from the canonical state store identity"
            )

        scheduler_ref = snapshot.scheduler.configuration_ref
        if (
            snapshot.scheduler.expected_configuration_ref is not None
            and scheduler_ref != snapshot.scheduler.expected_configuration_ref
        ):
            mismatches.add(
                "scheduler.configuration_ref: differs from the attested configuration"
            )

        entries = snapshot.tools.canonical_entries()
        for tool in snapshot.tools.entries:
            if (
                tool.expected_card_ref is not None
                and tool.card_ref != tool.expected_card_ref
            ):
                mismatches.add(
                    "tools.card_ref: at least one card differs from its attestation"
                )
        unauthorized_entries = [
            tool
            for tool in snapshot.tools.entries
            if tool.availability == "ACTIVE"
            and not _affordance_is_authorized(tool.affordance, snapshot.effects)
        ]
        if unauthorized_entries:
            mismatches.add(
                "tools.availability: an ACTIVE interface is outside the effect policy"
            )
        missing_executors = [
            tool
            for tool in snapshot.tools.entries
            if tool.availability == "ACTIVE"
            and tool.execution_mode == "SYNC"
            and tool.executor_ref is None
        ]
        if missing_executors:
            mismatches.add(
                "tools.executor_ref: an ACTIVE synchronous interface has no executor"
            )
        registry_ref = snapshot.tools.registry_ref
        if (
            snapshot.tools.expected_registry_ref is not None
            and registry_ref != snapshot.tools.expected_registry_ref
        ):
            mismatches.add(
                "tools.registry_ref: differs from the attested registry"
            )

        policy_ref = snapshot.effects.policy_ref
        if (
            snapshot.effects.expected_policy_ref is not None
            and policy_ref != snapshot.effects.expected_policy_ref
        ):
            mismatches.add(
                "effects.policy_ref: differs from the attested effect policy"
            )
        if (
            snapshot.effects.host_guarded_tools_enabled
            and snapshot.effects.host_catalog_ref is None
        ):
            mismatches.add(
                "effects.host_catalog_ref: absent while host-guarded tools are enabled"
            )

        model_payload: dict[str, object] = {
            "kind": binding.kind,
            "binding_ref": binding.binding_ref,
            "binding_file": binding_file_payload,
            "base_served_model": binding.base_served_model,
            "configured_served_model": binding.configured_served_model,
            "observed_served_model": binding.observed_served_model,
            "runtime_ref": binding.runtime_ref,
            "runtime_image": binding.runtime_image,
            "runtime_revision": binding.runtime_revision,
            "source_model": binding.source_model.payload(),
            "quantized_model": binding.quantized_model.payload(),
            "adapter": None if binding.adapter is None else binding.adapter.payload(),
            "selector_qualification_ref": binding.selector_qualification_ref,
        }
        scheduler_payload: dict[str, object] = {
            "implementation_ref": snapshot.scheduler.implementation_ref,
            "configuration_ref": scheduler_ref,
            "enabled": snapshot.scheduler.enabled,
            "life_loop_running": snapshot.scheduler.life_loop_running,
            "interval_seconds": float(snapshot.scheduler.interval_seconds),
            "max_steps_per_session": snapshot.scheduler.max_steps_per_session,
            "error": _redacted_error(snapshot.scheduler.error, "scheduler"),
        }
        memory_payload: dict[str, object] = {
            "canonical_writer_ref": snapshot.memory.canonical_writer_ref,
            "canonical_store_ref": snapshot.memory.canonical_store_ref,
            "cognee_is_canonical_writer": False,
            "projection_backend_refs": _sorted_unique(
                snapshot.memory.projection_backend_refs,
                "memory projection_backend_refs",
                maximum=32,
                validator=_ref,
            ),
            "pending_projections": snapshot.memory.pending_projections,
            "projection_error": _redacted_error(
                snapshot.memory.projection_error, "projection"
            ),
        }
        effects_payload = snapshot.effects.manifest_payload()
        assert effects_payload["policy_ref"] == policy_ref
        agent_payload: dict[str, object] = {
            "agent_ref": genesis.agent_ref,
            "genesis_contract": genesis.contract,
            "genesis_ref": genesis_ref,
        }
        guide_manifest_payload: dict[str, object] = {
            "document_id": GUIDE_DOCUMENT_ID,
            "contract": GUIDE_CONTRACT,
            "version": GUIDE_VERSION,
            "path": guide_payload["path"],
            "sha256": guide_payload["sha256"],
        }
        canonical_state_payload: dict[str, object] = {
            "contract": SUPERVISOR_STATE_CONTRACT,
            "store_identity_ref": state.store_identity_ref,
            "state_ref": state.head.state_ref,
            "revision": state.head.revision,
            "moving_origin_ordinal": state.head.moving_origin_ordinal,
            "last_event_ref": state.head.last_event_ref,
            "pending_ref": state.pending_ref,
        }
        time_payload: dict[str, object] = {
            "contract": temporal.contract,
            "sample_ref": temporal.sample_ref,
            "clock_anchor_ref": temporal.clock_anchor_ref,
            "trusted_utc": temporal.trusted_utc,
            "local_time": temporal.local_time,
            "local_timezone": temporal.local_timezone,
            "uncertainty_ms": float(temporal.uncertainty_ms),
            "jump_detected": temporal.jump_detected,
            "moving_origin_ordinal": temporal.moving_origin_ordinal,
        }
        tools_payload: dict[str, object] = {
            "registry_ref": registry_ref,
            "context_policy": TOOL_CONTEXT_POLICY,
            "entries": entries,
        }
        evidence_refs = _sorted_unique(
            snapshot.evidence_refs,
            "evidence_refs",
            maximum=256,
            validator=_ref,
        )
        verification_subjects = _derive_verification_subjects(
            {
                "agent": agent_payload,
                "guide": guide_manifest_payload,
                "schema_artifact": schema_payload,
                "model_binding": model_payload,
                "canonical_state": canonical_state_payload,
                "time": time_payload,
                "scheduler": scheduler_payload,
                "memory": memory_payload,
                "tools": tools_payload,
                "effects": effects_payload,
                "evidence_refs": evidence_refs,
            }
        )

        receipts = snapshot.verification.by_component()
        incomplete = REQUIRED_VERIFICATIONS - set(receipts)
        stale: set[str] = set()
        verification_operational_mismatch = False
        for component, receipt in receipts.items():
            if receipt.disposition == "UNAVAILABLE":
                incomplete.add(component)
                continue
            assert receipt.subject_ref is not None
            assert receipt.evidence_ref is not None
            component_mismatch = False
            if receipt.subject_ref != verification_subjects[component]:
                mismatches.add(
                    f"verification.{component}: subject differs from the captured component"
                )
                component_mismatch = True
            if receipt.evidence_ref not in evidence_refs:
                mismatches.add(
                    f"verification.{component}: evidence is absent from evidence_refs"
                )
                component_mismatch = True
            if receipt.disposition == "MISMATCH":
                mismatches.add(
                    f"{component}: external verification reported a mismatch"
                )
                component_mismatch = True
            elif receipt.disposition == "STALE":
                stale.add(component)
            if component_mismatch and component in _OPERATIONAL_VERIFICATIONS:
                verification_operational_mismatch = True

        incomplete_failures = {
            f"{component}: required verification is unavailable"
            for component in incomplete
        }
        stale_failures = {
            f"{component}: changed after the attested snapshot"
            for component in stale
        }
        if mismatches:
            integrity_status: IntegrityStatus = "MISMATCH"
        elif stale:
            integrity_status = "STALE"
        elif incomplete:
            integrity_status = "INCOMPLETE"
        else:
            integrity_status = "FRESH"
        integrity_failures = sorted(
            mismatches | incomplete_failures | stale_failures
        )
        if len(integrity_failures) > 64:
            raise CompositionInputError("integrity failures exceed the schema ceiling")

        operational_mismatch = verification_operational_mismatch or any(
            reason.split(":", 1)[0]
            in {
                "model_binding.configured_served_model",
                "model_binding.adapter.model_sha256",
                "model_binding.binding_ref",
                "model_binding.binding_file",
                "canonical_state.moving_origin_ordinal",
                "scheduler.enabled",
                "scheduler.configuration_ref",
                "memory.canonical_store_ref",
                "tools.availability",
                "tools.executor_ref",
                "tools.card_ref",
                "tools.registry_ref",
                "effects.policy_ref",
                "effects.host_catalog_ref",
            }
            for reason in mismatches
        )
        operational_status, operational_failures = _operational_status(
            snapshot,
            incomplete=incomplete,
            stale=stale,
            operational_mismatch=operational_mismatch,
        )

        unsigned: dict[str, object] = {
            "schema": MANIFEST_SCHEMA,
            "manifest_version": MANIFEST_VERSION,
            "hash_rule": MANIFEST_HASH_RULE,
            "schema_artifact": schema_payload,
            "generated_at_utc": temporal.trusted_utc,
            "integrity_status": integrity_status,
            "operational_status": operational_status,
            "integrity_failures": integrity_failures,
            "operational_failures": operational_failures,
            "agent": agent_payload,
            "guide": guide_manifest_payload,
            "model_binding": model_payload,
            "canonical_state": canonical_state_payload,
            "time": time_payload,
            "scheduler": scheduler_payload,
            "memory": memory_payload,
            "tools": tools_payload,
            "effects": effects_payload,
            "evidence_refs": evidence_refs,
        }
        manifest_ref = content_ref(unsigned)
        manifest = {**unsigned, "manifest_ref": manifest_ref}
        return JennyCompositionManifest(
            manifest,
            _build_token=_MANIFEST_BUILD_TOKEN,
        )


def _derive_verification_subjects(
    components: Mapping[str, object],
) -> dict[str, str]:
    expected = REQUIRED_VERIFICATIONS - {"snapshot_coherence"}
    if set(components) != expected:
        raise CompositionIntegrityError(
            "verification subject components do not match the v1 contract"
        )
    subjects = {
        component: content_ref(
            {
                "schema": "jenny2.composition-verification-subject.v1",
                "component": component,
                "payload": components[component],
            }
        )
        for component in sorted(components)
    }
    subjects["snapshot_coherence"] = content_ref(
        {
            "schema": "jenny2.composition-snapshot-coherence.v1",
            "subjects": subjects,
        }
    )
    return dict(sorted(subjects.items()))


def _verification_subjects_from_manifest(
    payload: Mapping[str, object],
) -> dict[str, str]:
    components = {
        component: payload[component]
        for component in REQUIRED_VERIFICATIONS - {"snapshot_coherence"}
    }
    return _derive_verification_subjects(components)


def _artifact_payload(
    artifact: LocalFileArtifactSnapshot,
    field: str,
    mismatches: set[str],
) -> tuple[dict[str, object], bytes]:
    observed_bytes = _read_regular_file(
        artifact.source_path, maximum_bytes=artifact.maximum_bytes
    )
    observed_sha256 = _raw_sha256(observed_bytes)
    if (
        artifact.expected_sha256 is not None
        and artifact.expected_sha256 != observed_sha256
    ):
        mismatches.add(f"{field}.sha256: bytes differ from the attested digest")
    return {
        "path": artifact.manifest_path,
        "sha256": observed_sha256,
    }, observed_bytes


def _verify_schema_contract(value: bytes, mismatches: set[str]) -> None:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        mismatches.add("schema_artifact: bytes are not a JSON schema")
        return
    properties = decoded.get("properties") if type(decoded) is dict else None
    schema_property = (
        properties.get("schema") if type(properties) is dict else None
    )
    version_property = (
        properties.get("manifest_version") if type(properties) is dict else None
    )
    hash_property = (
        properties.get("hash_rule") if type(properties) is dict else None
    )
    expected = (
        type(decoded) is dict
        and decoded.get("$id") == MANIFEST_SCHEMA_ID
        and type(schema_property) is dict
        and schema_property.get("const") == MANIFEST_SCHEMA
        and type(version_property) is dict
        and version_property.get("const") == MANIFEST_VERSION
        and type(hash_property) is dict
        and hash_property.get("const") == MANIFEST_HASH_RULE
    )
    if not expected:
        mismatches.add("schema_artifact: contract identity differs")


def _verify_guide_contract(value: bytes, mismatches: set[str]) -> None:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        mismatches.add("guide: bytes are not UTF-8")
        return
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    if not lines or lines[0].strip() != "---":
        mismatches.add("guide: front matter is absent")
        return
    front_matter_closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            front_matter_closed = True
            break
        if ":" not in line:
            continue
        key, item = line.split(":", 1)
        metadata[key.strip()] = item.strip()
    expected = {
        "document_id": GUIDE_DOCUMENT_ID,
        "guide_contract": GUIDE_CONTRACT,
        "guide_version": GUIDE_VERSION,
        "manifest_schema": "JENNY_2_LIVE_COMPOSITION_MANIFEST_V1.schema.json",
    }
    if not front_matter_closed or any(
        metadata.get(key) != item for key, item in expected.items()
    ):
        mismatches.add("guide: contract metadata differs")


def _verify_lora_binding_file(
    value: bytes,
    binding: ModelBindingSnapshot,
    mismatches: set[str],
) -> None:
    """Rejoin the loaded binding view to the exact content-addressed file."""

    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        mismatches.add("model_binding.binding_file: bytes are not JSON")
        return
    if type(decoded) is not dict:
        mismatches.add("model_binding.binding_file: payload is not an object")
        return
    observed_ref = decoded.get("binding_ref")
    unsigned = dict(decoded)
    unsigned.pop("binding_ref", None)
    if (
        observed_ref != binding.binding_ref
        or observed_ref != content_ref(unsigned)
    ):
        mismatches.add(
            "model_binding.binding_ref: differs from the binding file content"
        )
    assert binding.adapter is not None
    expected_fields: dict[str, object] = {
        "schema": "jenny2.sglang-lora-binding.v1",
        "base_served_model": binding.base_served_model,
        "served_model": binding.configured_served_model,
        "runtime_ref": binding.runtime_ref,
        "runtime_image": binding.runtime_image,
        "runtime_revision": binding.runtime_revision,
        "source_model_path": binding.source_model.path,
        "source_revision": binding.source_model.revision,
        "source_config_sha256": binding.source_model.config_sha256,
        "source_index_sha256": binding.source_model.index_sha256,
        "quantized_model_path": binding.quantized_model.path,
        "quantized_revision": binding.quantized_model.revision,
        "quantized_config_sha256": binding.quantized_model.config_sha256,
        "quantized_index_sha256": binding.quantized_model.index_sha256,
        "adapter_path": binding.adapter.path,
        "adapter_model_sha256": binding.adapter.model_sha256,
        "adapter_config_sha256": binding.adapter.config_sha256,
        "rank": binding.adapter.rank,
        "target_modules": list(binding.adapter.target_modules),
        "training_result_sha256": binding.adapter.training_result_sha256,
        "curriculum_manifest_sha256": (
            binding.adapter.curriculum_manifest_sha256
        ),
        "selector_qualification_ref": binding.selector_qualification_ref,
    }
    if any(decoded.get(key) != item for key, item in expected_fields.items()):
        mismatches.add(
            "model_binding.binding_file: loaded binding fields differ from its bytes"
        )


def _redacted_error(value: object | None, domain: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        private = (
            f"{type(value).__module__}.{type(value).__qualname__}:{value}"
        ).encode("utf-8", errors="backslashreplace")
    else:
        private = str(value).encode("utf-8", errors="backslashreplace")
    error_ref = "sha256:" + hashlib.sha256(private).hexdigest()
    return f"{domain} error details withheld; error_ref={error_ref}"


def _operational_status(
    snapshot: JennyCompositionSnapshot,
    *,
    incomplete: set[str],
    stale: set[str],
    operational_mismatch: bool,
) -> tuple[OperationalStatus, list[str]]:
    if incomplete & _OPERATIONAL_VERIFICATIONS:
        return (
            "UNKNOWN",
            ["operational status cannot be verified from the incomplete snapshot"],
        )
    if stale & _OPERATIONAL_VERIFICATIONS:
        return (
            "UNKNOWN",
            ["operational status cannot be verified from the stale snapshot"],
        )
    failures: set[str] = set()
    scheduler = snapshot.scheduler
    if scheduler.error is not None:
        failures.add("scheduler reported an error; details are hash-redacted")
    if scheduler.enabled and not scheduler.life_loop_running:
        failures.add("scheduler is enabled but the life loop is not running")
    if operational_mismatch:
        failures.add("an operational composition binding differs")
    if snapshot.memory.projection_error is not None:
        failures.add("memory projection reported an error; details are hash-redacted")
    degraded = sum(
        item.availability == "DEGRADED" for item in snapshot.tools.entries
    )
    if degraded:
        failures.add(f"{degraded} registered interface(s) are degraded")
    selectable = sum(
        item.availability == "ACTIVE"
        and _affordance_is_authorized(item.affordance, snapshot.effects)
        and (item.execution_mode != "SYNC" or item.executor_ref is not None)
        for item in snapshot.tools.entries
    )
    if selectable == 0:
        failures.add("no registered interface is currently selectable")
    unauthorized = sum(
        item.availability == "ACTIVE"
        and not _affordance_is_authorized(item.affordance, snapshot.effects)
        for item in snapshot.tools.entries
    )
    if unauthorized:
        failures.add(
            f"{unauthorized} active interface(s) are outside the effect policy"
        )
    unavailable_executors = sum(
        item.availability == "ACTIVE"
        and item.execution_mode == "SYNC"
        and item.executor_ref is None
        for item in snapshot.tools.entries
    )
    if unavailable_executors:
        failures.add(
            f"{unavailable_executors} active synchronous interface(s) lack an executor reference"
        )
    if (
        snapshot.effects.host_guarded_tools_enabled
        and snapshot.effects.host_catalog_ref is None
    ):
        failures.add("host-guarded tools are enabled without a catalog reference")
    ordered = sorted(failures)
    if not scheduler.enabled:
        if "scheduler is intentionally disabled" not in failures:
            ordered.append("scheduler is intentionally disabled")
            ordered.sort()
        return "STOPPED", ordered
    if ordered:
        return "DEGRADED", ordered
    return "READY", []


def _affordance_is_authorized(
    affordance: Affordance, effects: EffectPolicySnapshot
) -> bool:
    if affordance.authorization_mode == "HOST_GUARDED":
        return (
            effects.host_guarded_tools_enabled
            and affordance.permission_scope in effects.allowed_host_guarded_scopes
        )
    if affordance.external_effect:
        return (
            effects.general_external_effects_enabled
            or affordance.permission_scope in effects.allowed_read_only_scopes
        )
    return affordance.permission_scope in effects.allowed_internal_scopes


def _project_validated_payload(payload: dict[str, object]) -> dict[str, object]:
    """Project only a payload which has passed the builder's full checks."""

    observed_ref = payload.get("manifest_ref")
    unsigned = dict(payload)
    unsigned.pop("manifest_ref", None)
    if observed_ref != content_ref(unsigned):
        raise CompositionIntegrityError("manifest_ref differs from canonical bytes")
    integrity = payload.get("integrity_status")
    if integrity not in ("FRESH", "STALE", "MISMATCH", "INCOMPLETE"):
        raise CompositionIntegrityError("manifest integrity status is unsupported")
    failures = payload.get("integrity_failures")
    if type(failures) is not list or any(type(item) is not str for item in failures):
        raise CompositionIntegrityError("manifest integrity failures are malformed")

    self_system: dict[str, object] = {
        "agent_ref": payload["agent"]["agent_ref"],  # type: ignore[index]
        "composition_ref": observed_ref,
        "composition_integrity": integrity,
        "operational_status": payload["operational_status"],
        "operational_failures": payload["operational_failures"],
        "claims_current": integrity == "FRESH",
        "refresh_required": integrity != "FRESH",
    }
    world_interfaces: dict[str, object] = {
        "composition_ref": observed_ref,
        "composition_integrity": integrity,
        "operational_status": payload["operational_status"],
        "operational_failures": payload["operational_failures"],
        "claims_current": integrity == "FRESH",
        "refresh_required": integrity != "FRESH",
        "entries": [],
    }
    if integrity != "FRESH":
        self_system["integrity_failures"] = failures
        world_interfaces["integrity_failures"] = failures
        return {
            "SELF": {"system": self_system},
            "WORLD": {"available_interfaces": world_interfaces},
        }

    guide = payload["guide"]  # type: ignore[assignment]
    model = payload["model_binding"]  # type: ignore[assignment]
    state = payload["canonical_state"]  # type: ignore[assignment]
    temporal = payload["time"]  # type: ignore[assignment]
    tools = payload["tools"]  # type: ignore[assignment]
    effects = payload["effects"]  # type: ignore[assignment]
    guide_ref = "sha256:" + guide["sha256"]  # type: ignore[index,operator]
    self_system.update(
        {
            "guide": {
                "document_id": guide["document_id"],  # type: ignore[index]
                "version": guide["version"],  # type: ignore[index]
                "sha256": guide["sha256"],  # type: ignore[index]
            },
            "cortex": {
                "served_model": model["observed_served_model"],  # type: ignore[index]
                "binding_ref": model["binding_ref"],  # type: ignore[index]
            },
            "state": {
                "state_ref": state["state_ref"],  # type: ignore[index]
                "revision": state["revision"],  # type: ignore[index]
                "moving_origin_ordinal": state["moving_origin_ordinal"],  # type: ignore[index]
            },
            "time_sample_ref": temporal["sample_ref"],  # type: ignore[index]
            "tool_registry_ref": tools["registry_ref"],  # type: ignore[index]
            "effect_policy_ref": effects["policy_ref"],  # type: ignore[index]
            "full_guide_ref": guide_ref,
        }
    )
    compact_entries: list[dict[str, object]] = []
    for entry in tools["entries"]:  # type: ignore[index,union-attr]
        if entry["availability"] != "ACTIVE":
            continue
        compact_entries.append(
            {
                "affordance_id": entry["affordance_id"],
                "card_ref": entry["card_ref"],
                "full_definition_ref": entry["full_definition_ref"],
                "executor_ref": entry["executor_ref"],
                "observable_source_ref": entry["observable_source_ref"],
                "permission_scope": entry["permission_scope"],
                "external_effect": entry["external_effect"],
                "authorization_mode": entry["authorization_mode"],
                "execution_mode": entry["execution_mode"],
                "availability": entry["availability"],
                "selectable": True,
            }
        )
    world_interfaces.update(
        {
            "tool_registry_ref": tools["registry_ref"],  # type: ignore[index]
            "effect_policy_ref": effects["policy_ref"],  # type: ignore[index]
            "entries": compact_entries,
            "effect_boundary": {
                "general_external_effects_enabled": effects[
                    "general_external_effects_enabled"
                ],  # type: ignore[index]
                "allowed_read_only_scopes": effects[
                    "allowed_read_only_scopes"
                ],  # type: ignore[index]
                "allowed_host_guarded_scopes": effects[
                    "allowed_host_guarded_scopes"
                ],  # type: ignore[index]
                "host_guarded_tools_enabled": effects[
                    "host_guarded_tools_enabled"
                ],  # type: ignore[index]
                "host_catalog_ref": effects["host_catalog_ref"],  # type: ignore[index]
            },
        }
    )
    return {
        "SELF": {"system": self_system},
        "WORLD": {"available_interfaces": world_interfaces},
    }


def project_self_world(
    manifest: JennyCompositionManifest,
) -> dict[str, object]:
    """Return the bounded projection of a builder-validated manifest.

    Arbitrary dictionaries are intentionally rejected.  A recomputable hash
    is an integrity checksum, not an attestation that a caller-provided object
    passed the file, runtime, and cross-field checks needed for ``FRESH``.
    """

    if not isinstance(manifest, JennyCompositionManifest):
        raise CompositionIntegrityError(
            "SELF/WORLD projection requires a builder-validated manifest"
        )
    return manifest.self_world_projection()


def verification_subjects(
    manifest: JennyCompositionManifest,
) -> dict[str, str]:
    """Return the exact subjects an external snapshot validator must attest.

    An integration can build an ``INCOMPLETE`` draft while holding the shared
    operation lock, validate these references, add evidence-bound receipts,
    and rebuild from the unchanged snapshot.  Aggregate status is excluded
    from the references, so that two-pass process is stable.
    """

    if not isinstance(manifest, JennyCompositionManifest):
        raise CompositionIntegrityError(
            "verification subjects require a builder-validated manifest"
        )
    return _verification_subjects_from_manifest(manifest.to_dict())


def build_jenny_composition_manifest(
    snapshot: JennyCompositionSnapshot,
) -> JennyCompositionManifest:
    return JennyCompositionManifestBuilder().build(snapshot)


build_live_composition_manifest = build_jenny_composition_manifest


__all__ = [
    "AFFORDANCE_CARD_SCHEMA",
    "AdapterSnapshot",
    "AffordanceSnapshot",
    "CanonicalMemorySnapshot",
    "CanonicalStateSnapshot",
    "CompositionArtifactError",
    "CompositionInputError",
    "CompositionIntegrityError",
    "CompositionVerification",
    "EFFECT_POLICY_SCHEMA",
    "EffectPolicySnapshot",
    "GUIDE_CONTRACT",
    "GUIDE_DOCUMENT_ID",
    "GUIDE_VERSION",
    "JennyCompositionManifest",
    "JennyCompositionManifestBuilder",
    "JennyCompositionSnapshot",
    "LocalFileArtifactSnapshot",
    "MANIFEST_HASH_RULE",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_ID",
    "MANIFEST_VERSION",
    "ModelArtifactSnapshot",
    "ModelBindingSnapshot",
    "REQUIRED_VERIFICATIONS",
    "SCHEDULER_CONFIGURATION_SCHEMA",
    "SchedulerSnapshot",
    "TOOL_CONTEXT_POLICY",
    "TOOL_REGISTRY_SCHEMA",
    "ToolRegistrySnapshot",
    "VerificationDisposition",
    "VerificationReceipt",
    "build_jenny_composition_manifest",
    "build_live_composition_manifest",
    "canonical_json",
    "canonical_json_bytes",
    "content_ref",
    "project_self_world",
    "verification_subjects",
]
