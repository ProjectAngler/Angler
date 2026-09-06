from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from angler.runtime.jenny_composition import (
    AdapterSnapshot,
    AffordanceSnapshot,
    CanonicalMemorySnapshot,
    CanonicalStateSnapshot,
    CompositionIntegrityError,
    CompositionVerification,
    EffectPolicySnapshot,
    JennyCompositionManifest,
    JennyCompositionSnapshot,
    LocalFileArtifactSnapshot,
    MANIFEST_HASH_RULE,
    ModelArtifactSnapshot,
    ModelBindingSnapshot,
    REQUIRED_VERIFICATIONS,
    SchedulerSnapshot,
    ToolRegistrySnapshot,
    VerificationReceipt,
    build_jenny_composition_manifest,
    canonical_json_bytes,
    content_ref,
    project_self_world,
    verification_subjects,
)
from angler.runtime.jenny_genesis import JennyGenesis
from angler.runtime.persistent_autonomy import Affordance, SupervisorStateHead
from angler.runtime.temporal_v2 import TEMPORAL_NOW_CONTRACT, TemporalNow


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "docs/reports/JENNY_2_LIVE_COMPOSITION_MANIFEST_V1.schema.json"
GUIDE_PATH = ROOT / "docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ref(label: str) -> str:
    return "sha256:" + _digest(label)


def _verification_evidence_ref(component: str) -> str:
    return _ref(f"verification-evidence:{component}")


def _artifact(
    path: Path, public_path: str, *, expected: bool = True
) -> LocalFileArtifactSnapshot:
    if expected:
        return LocalFileArtifactSnapshot.capture(path, manifest_path=public_path)
    return LocalFileArtifactSnapshot(path, public_path)


def _model(binding_file: LocalFileArtifactSnapshot) -> ModelBindingSnapshot:
    return ModelBindingSnapshot(
        kind="LORA",
        binding_ref=_ref("binding"),
        binding_file=binding_file,
        base_served_model="jenny-qwen3.8-27b",
        configured_served_model=(
            "jenny-qwen3.8-27b--lora-sha256-" + _digest("adapter")
        ),
        observed_served_model=(
            "jenny-qwen3.8-27b--lora-sha256-" + _digest("adapter")
        ),
        runtime_ref=_ref("runtime"),
        runtime_image="lmsysorg/sglang@sha256:" + _digest("runtime-image"),
        runtime_revision="5f55db35e926d50676f75b812640ea2410b0fe0e",
        source_model=ModelArtifactSnapshot(
            path="/models/Qwen3.8-27B",
            revision="e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c",
            config_sha256=_digest("source-config"),
            index_sha256=_digest("source-index"),
        ),
        quantized_model=ModelArtifactSnapshot(
            path="/models/Qwen3.8-27B-NVFP4",
            revision="319f741cce68d7914884900c138a1fbb70a42f30",
            config_sha256=_digest("quantized-config"),
            index_sha256=_digest("quantized-index"),
        ),
        adapter=AdapterSnapshot(
            path="/models/adapters/jenny-r5",
            model_sha256=_digest("adapter"),
            config_sha256=_digest("adapter-config"),
            rank=16,
            target_modules=("v_proj", "q_proj", "k_proj"),
            training_result_sha256=_digest("training-result"),
            curriculum_manifest_sha256=_digest("curriculum"),
        ),
        selector_qualification_ref=None,
    )


def _binding_file_payload(model: ModelBindingSnapshot) -> dict[str, object]:
    assert model.adapter is not None
    unsigned: dict[str, object] = {
        "schema": "jenny2.sglang-lora-binding.v1",
        "endpoint": "http://127.0.0.1:30000/v1",
        "base_served_model": model.base_served_model,
        "served_model": model.configured_served_model,
        "runtime_ref": model.runtime_ref,
        "runtime_image": model.runtime_image,
        "runtime_revision": model.runtime_revision,
        "source_model_path": model.source_model.path,
        "source_revision": model.source_model.revision,
        "source_config_sha256": model.source_model.config_sha256,
        "source_index_sha256": model.source_model.index_sha256,
        "quantized_model_path": model.quantized_model.path,
        "quantized_revision": model.quantized_model.revision,
        "quantized_config_sha256": model.quantized_model.config_sha256,
        "quantized_index_sha256": model.quantized_model.index_sha256,
        "adapter_path": model.adapter.path,
        "adapter_model_sha256": model.adapter.model_sha256,
        "adapter_config_sha256": model.adapter.config_sha256,
        "rank": model.adapter.rank,
        "target_modules": list(model.adapter.target_modules),
        "training_result_sha256": model.adapter.training_result_sha256,
        "curriculum_manifest_sha256": model.adapter.curriculum_manifest_sha256,
        "selector_qualification_ref": model.selector_qualification_ref,
    }
    return {**unsigned, "binding_ref": content_ref(unsigned)}


def _tools(*, secret_description: str | None = None) -> ToolRegistrySnapshot:
    web = AffordanceSnapshot(
        affordance=Affordance(
            affordance_id="tool.web_read",
            disposition="ACT",
            description="Read one bounded public HTTPS resource.",
            permission_scope="external.readonly.web",
            external_effect=True,
        ),
        full_definition_ref=_ref("web-full-definition"),
        executor_ref=_ref("web-executor"),
        observable_source_ref=_ref("web-source"),
        execution_mode="SYNC",
    )
    cortex = AffordanceSnapshot(
        affordance=Affordance(
            affordance_id="cortex.respond",
            disposition="ACT",
            description=(
                secret_description
                if secret_description is not None
                else "Respond through the frozen public cortex."
            ),
            permission_scope="internal.cognition",
        ),
        full_definition_ref=_ref("cortex-full-definition"),
        executor_ref=_ref("cortex-executor"),
        observable_source_ref=None,
        execution_mode="SYNC",
    )
    # Deliberately noncanonical input order; the builder owns registry order.
    return ToolRegistrySnapshot(entries=(web, cortex))


def _snapshot(
    tmp_path: Path,
    *,
    guide_artifact: LocalFileArtifactSnapshot | None = None,
    verification: CompositionVerification | None = None,
    scheduler_error: object | None = None,
    memory_error: object | None = None,
    secret_description: str | None = None,
) -> JennyCompositionSnapshot:
    binding_path = tmp_path / "binding.json"
    public_binding_path = "runtime/bindings/jenny-r5.json"
    provisional_binding_file = LocalFileArtifactSnapshot(
        binding_path,
        public_binding_path,
    )
    model = _model(provisional_binding_file)
    binding_payload = _binding_file_payload(model)
    binding_path.write_bytes(canonical_json_bytes(binding_payload))
    binding_file = _artifact(
        binding_path,
        public_binding_path,
    )
    model = replace(
        model,
        binding_ref=binding_payload["binding_ref"],
        binding_file=binding_file,
    )
    store_ref = _ref("canonical-store-identity-without-a-path")
    state_head = SupervisorStateHead(
        state_ref=_ref("state-7"),
        moving_origin_ordinal=7,
        last_event_ref=_ref("event-7"),
        scheduler_enabled=True,
        revision=11,
    )
    temporal = TemporalNow(
        contract=TEMPORAL_NOW_CONTRACT,
        trusted_utc="2026-09-03T17:00:00.000000Z",
        local_time="2026-09-03T13:00:00.000000-04:00",
        local_timezone="America/New_York",
        local_utc_offset_seconds=-14_400,
        monotonic_ns=123_000_000,
        clock_anchor_ref=_ref("clock-anchor"),
        uncertainty_ms=1.0,
        jump_detected=False,
        wall_elapsed_ms=12_000.0,
        monotonic_elapsed_ms=12_000.0,
        moving_origin_ordinal=7,
    )
    snapshot = JennyCompositionSnapshot(
        genesis=JennyGenesis.owner_approved(
            created_at_utc="2026-09-02T00:00:00.000000Z"
        ),
        guide_artifact=(
            guide_artifact
            if guide_artifact is not None
            else _artifact(
                GUIDE_PATH,
                "docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md",
            )
        ),
        schema_artifact=_artifact(
            SCHEMA_PATH,
            "docs/reports/JENNY_2_LIVE_COMPOSITION_MANIFEST_V1.schema.json",
        ),
        model_binding=model,
        canonical_state=CanonicalStateSnapshot(
            head=state_head,
            store_identity_ref=store_ref,
            pending_ref=None,
        ),
        time=temporal,
        scheduler=SchedulerSnapshot(
            implementation_ref=_ref("scheduler-implementation"),
            enabled=True,
            life_loop_running=True,
            interval_seconds=30.0,
            max_steps_per_session=10_000,
            error=scheduler_error,
        ),
        memory=CanonicalMemorySnapshot(
            canonical_writer_ref=_ref("canonical-writer"),
            canonical_store_ref=store_ref,
            projection_backend_refs=(
                _ref("projection-z"),
                _ref("projection-a"),
            ),
            pending_projections=0,
            projection_error=memory_error,
        ),
        tools=_tools(secret_description=secret_description),
        effects=EffectPolicySnapshot(
            general_external_effects_enabled=False,
            allowed_internal_scopes=("internal.cognition",),
            allowed_read_only_scopes=(
                "external.readonly.web",
                "external.readonly.library",
            ),
            allowed_host_guarded_scopes=(
                "external.openclaw.guarded",
                "external.workspace.guarded",
            ),
            host_guarded_tools_enabled=False,
            host_catalog_ref=None,
        ),
        evidence_refs=(
            _ref("evidence-z"),
            _ref("evidence-a"),
            *(
                _verification_evidence_ref(component)
                for component in sorted(REQUIRED_VERIFICATIONS)
            ),
        ),
        verification=(
            verification
            if verification is not None
            else CompositionVerification()
        ),
    )
    return snapshot if verification is not None else _attest(snapshot)


def _verification_for(
    snapshot: JennyCompositionSnapshot,
    *,
    stale: frozenset[str] = frozenset(),
    mismatch: frozenset[str] = frozenset(),
    omit: frozenset[str] = frozenset(),
) -> CompositionVerification:
    draft = build_jenny_composition_manifest(
        replace(snapshot, verification=CompositionVerification())
    )
    subjects = verification_subjects(draft)
    receipts = []
    for component in sorted(REQUIRED_VERIFICATIONS - omit):
        disposition = (
            "MISMATCH"
            if component in mismatch
            else "STALE"
            if component in stale
            else "EXACT"
        )
        receipts.append(
            VerificationReceipt(
                component=component,
                disposition=disposition,
                subject_ref=subjects[component],
                evidence_ref=_verification_evidence_ref(component),
            )
        )
    return CompositionVerification(tuple(receipts))


def _attest(
    snapshot: JennyCompositionSnapshot,
    *,
    stale: frozenset[str] = frozenset(),
    mismatch: frozenset[str] = frozenset(),
    omit: frozenset[str] = frozenset(),
) -> JennyCompositionSnapshot:
    return replace(
        snapshot,
        verification=_verification_for(
            snapshot,
            stale=stale,
            mismatch=mismatch,
            omit=omit,
        ),
    )


def _validate_schema(payload: dict[str, object]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(payload) == set(schema["required"])
    assert payload["schema"] == schema["properties"]["schema"]["const"]
    assert payload["manifest_version"] == schema["properties"][
        "manifest_version"
    ]["const"]
    assert payload["hash_rule"] == schema["properties"]["hash_rule"]["const"]
    assert payload["integrity_status"] in schema["properties"][
        "integrity_status"
    ]["enum"]
    assert payload["operational_status"] in schema["properties"][
        "operational_status"
    ]["enum"]
    assert set(payload["agent"]) == set(schema["properties"]["agent"]["required"])
    assert set(payload["guide"]) == set(schema["properties"]["guide"]["required"])
    assert set(payload["model_binding"]) == set(
        schema["properties"]["model_binding"]["required"]
    )
    assert set(payload["canonical_state"]) == set(
        schema["properties"]["canonical_state"]["required"]
    )
    assert set(payload["time"]) == set(schema["properties"]["time"]["required"])
    assert set(payload["scheduler"]) == set(
        schema["properties"]["scheduler"]["required"]
    )
    assert set(payload["memory"]) == set(schema["properties"]["memory"]["required"])
    assert set(payload["tools"]) == set(schema["properties"]["tools"]["required"])
    assert set(payload["effects"]) == set(
        schema["properties"]["effects"]["required"]
    )
    for entry in payload["tools"]["entries"]:
        assert set(entry) == set(schema["$defs"]["tool_entry"]["required"])
    if payload["model_binding"]["kind"] == "LORA":
        assert payload["model_binding"]["binding_ref"] is not None
        assert payload["model_binding"]["binding_file"] is not None
        assert payload["model_binding"]["adapter"] is not None
    else:
        assert payload["model_binding"]["kind"] == "BASE_CONTROL"
        assert payload["model_binding"]["binding_ref"] is None
        assert payload["model_binding"]["binding_file"] is None
        assert payload["model_binding"]["adapter"] is None
    if payload["integrity_status"] == "FRESH":
        assert payload["integrity_failures"] == []
    else:
        assert payload["integrity_failures"]


def test_fresh_manifest_is_schema_valid_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)

    first = build_jenny_composition_manifest(snapshot)
    second = build_jenny_composition_manifest(snapshot)
    payload = first.to_dict()

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.manifest_ref == second.manifest_ref
    assert payload["integrity_status"] == "FRESH"
    assert payload["operational_status"] == "READY"
    assert payload["integrity_failures"] == []
    assert payload["operational_failures"] == []
    assert payload["hash_rule"] == MANIFEST_HASH_RULE
    unsigned = dict(payload)
    observed_ref = unsigned.pop("manifest_ref")
    assert observed_ref == content_ref(unsigned)
    with pytest.raises(ValueError, match="NaN and Infinity"):
        canonical_json_bytes({"invalid": float("nan")})
    _validate_schema(payload)

    projection = first.self_world_projection()
    system = projection["SELF"]["system"]
    interfaces = projection["WORLD"]["available_interfaces"]
    assert system["claims_current"] is True
    assert system["composition_ref"] == first.manifest_ref
    assert system["cortex"]["served_model"] == (
        snapshot.model_binding.observed_served_model
    )
    assert system["state"] == {
        "state_ref": snapshot.canonical_state.head.state_ref,
        "revision": 11,
        "moving_origin_ordinal": 7,
    }
    assert [entry["affordance_id"] for entry in interfaces["entries"]] == [
        "cortex.respond",
        "tool.web_read",
    ]
    assert all(entry["selectable"] is True for entry in interfaces["entries"])


def test_component_changes_change_manifest_identity(tmp_path: Path) -> None:
    original = _snapshot(tmp_path)
    first = build_jenny_composition_manifest(original)
    changed_head = replace(
        original.canonical_state.head,
        state_ref=_ref("state-8"),
        revision=12,
    )
    changed = replace(
        original,
        canonical_state=replace(original.canonical_state, head=changed_head),
    )

    second = build_jenny_composition_manifest(_attest(changed))

    assert second.manifest_ref != first.manifest_ref
    assert second["canonical_state"]["state_ref"] == _ref("state-8")


def test_base_control_conditional_is_emitted_without_adapter_binding(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    base = replace(
        snapshot.model_binding,
        kind="BASE_CONTROL",
        binding_ref=None,
        binding_file=None,
        configured_served_model=snapshot.model_binding.base_served_model,
        observed_served_model=snapshot.model_binding.base_served_model,
        adapter=None,
    )

    manifest = build_jenny_composition_manifest(
        _attest(replace(snapshot, model_binding=base))
    )

    assert manifest["integrity_status"] == "FRESH"
    assert manifest["model_binding"]["binding_ref"] is None
    assert manifest["model_binding"]["binding_file"] is None
    assert manifest["model_binding"]["adapter"] is None
    assert manifest.self_world["SELF"]["system"]["cortex"]["binding_ref"] is None
    _validate_schema(manifest.to_dict())


def test_set_inputs_and_tools_are_sorted_without_changing_identity(
    tmp_path: Path,
) -> None:
    original = _snapshot(tmp_path)
    reordered = replace(
        original,
        tools=ToolRegistrySnapshot(entries=tuple(reversed(original.tools.entries))),
        effects=replace(
            original.effects,
            allowed_read_only_scopes=tuple(
                reversed(original.effects.allowed_read_only_scopes)
            ),
            allowed_host_guarded_scopes=tuple(
                reversed(original.effects.allowed_host_guarded_scopes)
            ),
        ),
        evidence_refs=tuple(reversed(original.evidence_refs)),
        memory=replace(
            original.memory,
            projection_backend_refs=tuple(
                reversed(original.memory.projection_backend_refs)
            ),
        ),
    )

    first = build_jenny_composition_manifest(original)
    second = build_jenny_composition_manifest(reordered)
    payload = first.to_dict()

    assert second.manifest_ref == first.manifest_ref
    assert [entry["affordance_id"] for entry in payload["tools"]["entries"]] == [
        "cortex.respond",
        "tool.web_read",
    ]
    assert payload["evidence_refs"] == sorted(payload["evidence_refs"])
    assert payload["memory"]["projection_backend_refs"] == sorted(
        payload["memory"]["projection_backend_refs"]
    )
    assert payload["model_binding"]["adapter"]["target_modules"] == [
        "k_proj",
        "q_proj",
        "v_proj",
    ]


def test_mismatch_is_fail_visible_and_suppresses_present_tense_projection(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    mismatch = replace(
        snapshot,
        model_binding=replace(
            snapshot.model_binding,
            observed_served_model="jenny-qwen3.8-27b",
        ),
    )

    manifest = build_jenny_composition_manifest(_attest(mismatch))
    payload = manifest.to_dict()

    assert payload["integrity_status"] == "MISMATCH"
    assert payload["operational_status"] == "DEGRADED"
    assert any(
        reason.startswith("model_binding.configured_served_model:")
        for reason in payload["integrity_failures"]
    )
    projection = manifest.self_world_projection()
    system = projection["SELF"]["system"]
    available = projection["WORLD"]["available_interfaces"]
    assert system["claims_current"] is False
    assert system["refresh_required"] is True
    assert "cortex" not in system
    assert "state" not in system
    assert available["entries"] == []
    assert "tool_registry_ref" not in available
    _validate_schema(payload)


def test_state_and_trusted_time_ordinal_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    mismatch = replace(
        snapshot,
        time=replace(snapshot.time, moving_origin_ordinal=8),
    )

    manifest = build_jenny_composition_manifest(_attest(mismatch))

    assert manifest["integrity_status"] == "MISMATCH"
    assert manifest["operational_status"] == "DEGRADED"
    assert (
        "canonical_state.moving_origin_ordinal: differs from the trusted-time sample"
        in manifest["integrity_failures"]
    )
    assert manifest.self_world["WORLD"]["available_interfaces"]["entries"] == []


def test_stale_and_incomplete_states_are_derived_from_verification(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    stale = _attest(
        snapshot,
        stale=frozenset({"canonical_state"}),
    )
    stale_manifest = build_jenny_composition_manifest(stale)
    assert stale_manifest["integrity_status"] == "STALE"
    assert stale_manifest["integrity_failures"] == [
        "canonical_state: changed after the attested snapshot"
    ]

    incomplete = _attest(
        snapshot,
        omit=frozenset({"tools"}),
    )
    incomplete_manifest = build_jenny_composition_manifest(incomplete)
    assert incomplete_manifest["integrity_status"] == "INCOMPLETE"
    assert incomplete_manifest["operational_status"] == "UNKNOWN"
    assert incomplete_manifest["integrity_failures"] == [
        "tools: required verification is unavailable"
    ]
    _validate_schema(stale_manifest.to_dict())
    _validate_schema(incomplete_manifest.to_dict())


def test_raw_errors_secrets_descriptions_and_absolute_paths_do_not_leak(
    tmp_path: Path,
) -> None:
    secret = "never-print-this-bearer-token"
    private_state_path = f"/private/canonical/{secret}/state.sqlite3"
    private_guide = tmp_path / f"{secret}-guide.md"
    shutil.copyfile(GUIDE_PATH, private_guide)
    snapshot = _snapshot(
        tmp_path,
        guide_artifact=_artifact(
            private_guide,
            "docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md",
        ),
        scheduler_error=f"failure at {private_state_path}",
        memory_error=f"Authorization: Bearer {secret}",
        secret_description=f"private description {secret}",
    )

    manifest = build_jenny_composition_manifest(snapshot)
    serialized = manifest.canonical_json()
    cognitive = json.dumps(manifest.self_world_projection(), sort_keys=True)

    assert secret not in serialized
    assert private_state_path not in serialized
    assert str(private_guide) not in serialized
    assert "error_ref=sha256:" in serialized
    assert secret not in cognitive
    assert "/models/" not in cognitive
    assert "docs/reports/" not in cognitive
    assert "path" not in cognitive.lower()


def test_guide_bytes_change_manifest_and_pinned_bytes_mismatch(tmp_path: Path) -> None:
    copied_guide = tmp_path / "guide.md"
    shutil.copyfile(GUIDE_PATH, copied_guide)
    floating = _artifact(
        copied_guide,
        "docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md",
        expected=False,
    )
    first = build_jenny_composition_manifest(
        _snapshot(tmp_path, guide_artifact=floating)
    )
    pinned = LocalFileArtifactSnapshot.capture(
        copied_guide,
        manifest_path="docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md",
    )

    copied_guide.write_text(
        copied_guide.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    second = build_jenny_composition_manifest(
        _snapshot(tmp_path, guide_artifact=floating)
    )
    mismatched = build_jenny_composition_manifest(
        _snapshot(tmp_path, guide_artifact=pinned)
    )

    assert second.manifest_ref != first.manifest_ref
    assert second["guide"]["sha256"] != first["guide"]["sha256"]
    assert second["integrity_status"] == "FRESH"
    assert mismatched["integrity_status"] == "MISMATCH"
    assert "guide.sha256: bytes differ from the attested digest" in (
        mismatched["integrity_failures"]
    )


def test_fresh_projection_cannot_be_forged_from_an_arbitrary_mapping(
    tmp_path: Path,
) -> None:
    manifest = build_jenny_composition_manifest(_snapshot(tmp_path))
    payload = manifest.to_dict()

    with pytest.raises(CompositionIntegrityError, match="validated builder"):
        JennyCompositionManifest(payload)
    with pytest.raises(CompositionIntegrityError, match="builder-validated"):
        project_self_world(payload)  # type: ignore[arg-type]
    with pytest.raises(CompositionIntegrityError, match="builder-validated"):
        verification_subjects(payload)  # type: ignore[arg-type]


def test_verification_receipts_are_bound_to_subject_and_evidence(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    receipts = list(snapshot.verification.receipts)
    tools_index = next(
        index
        for index, receipt in enumerate(receipts)
        if receipt.component == "tools"
    )
    receipts[tools_index] = replace(
        receipts[tools_index],
        subject_ref=_ref("forged-tools-subject"),
    )
    forged_subject = replace(
        snapshot,
        verification=CompositionVerification(tuple(receipts)),
    )

    subject_manifest = build_jenny_composition_manifest(forged_subject)
    assert subject_manifest["integrity_status"] == "MISMATCH"
    assert (
        "verification.tools: subject differs from the captured component"
        in subject_manifest["integrity_failures"]
    )

    missing_evidence = replace(
        snapshot,
        evidence_refs=tuple(
            item
            for item in snapshot.evidence_refs
            if item != _verification_evidence_ref("tools")
        ),
    )
    evidence_manifest = build_jenny_composition_manifest(missing_evidence)
    assert evidence_manifest["integrity_status"] == "MISMATCH"
    assert (
        "verification.tools: evidence is absent from evidence_refs"
        in evidence_manifest["integrity_failures"]
    )


def test_owner_genesis_and_loaded_adapter_identity_are_rechecked(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    altered_genesis = replace(
        snapshot,
        genesis=replace(
            snapshot.genesis,
            archive_path=r"C:\\not-the-owner-pinned-archive.tar.zst",
        ),
    )
    genesis_manifest = build_jenny_composition_manifest(_attest(altered_genesis))
    assert genesis_manifest["integrity_status"] == "MISMATCH"
    assert (
        "agent.genesis_ref: differs from the owner-pinned genesis"
        in genesis_manifest["integrity_failures"]
    )

    assert snapshot.model_binding.adapter is not None
    altered_binding = replace(
        snapshot,
        model_binding=replace(
            snapshot.model_binding,
            adapter=replace(
                snapshot.model_binding.adapter,
                model_sha256=_digest("different-adapter"),
            ),
        ),
    )
    binding_manifest = build_jenny_composition_manifest(_attest(altered_binding))
    assert binding_manifest["integrity_status"] == "MISMATCH"
    assert (
        "model_binding.adapter.model_sha256: differs from the configured served model"
        in binding_manifest["integrity_failures"]
    )
    assert (
        "model_binding.binding_file: loaded binding fields differ from its bytes"
        in binding_manifest["integrity_failures"]
    )


def test_choice_projection_marks_degraded_tools_and_hides_unauthorized_active_tools(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    degraded_entries = tuple(
        replace(entry, availability="DEGRADED")
        if entry.affordance.affordance_id == "tool.web_read"
        else entry
        for entry in snapshot.tools.entries
    )
    degraded = _attest(
        replace(snapshot, tools=ToolRegistrySnapshot(degraded_entries))
    )
    degraded_manifest = build_jenny_composition_manifest(degraded)
    interfaces = degraded_manifest.self_world["WORLD"]["available_interfaces"]
    by_id = {entry["affordance_id"]: entry for entry in interfaces["entries"]}

    assert degraded_manifest["integrity_status"] == "FRESH"
    assert degraded_manifest["operational_status"] == "DEGRADED"
    assert by_id["cortex.respond"]["selectable"] is True
    assert "tool.web_read" not in by_id
    assert interfaces["operational_failures"] == degraded_manifest[
        "operational_failures"
    ]

    unauthorized = _attest(
        replace(
            snapshot,
            effects=replace(
                snapshot.effects,
                allowed_read_only_scopes=("external.readonly.library",),
            ),
        )
    )
    unauthorized_manifest = build_jenny_composition_manifest(unauthorized)
    assert unauthorized_manifest["integrity_status"] == "MISMATCH"
    assert (
        "tools.availability: an ACTIVE interface is outside the effect policy"
        in unauthorized_manifest["integrity_failures"]
    )
    assert unauthorized_manifest.self_world["WORLD"]["available_interfaces"][
        "entries"
    ] == []

    no_choice_entries = tuple(
        replace(entry, availability="INACTIVE")
        for entry in snapshot.tools.entries
    )
    no_choice = _attest(
        replace(snapshot, tools=ToolRegistrySnapshot(no_choice_entries))
    )
    no_choice_manifest = build_jenny_composition_manifest(no_choice)
    assert no_choice_manifest["integrity_status"] == "FRESH"
    assert no_choice_manifest["operational_status"] == "DEGRADED"
    assert "no registered interface is currently selectable" in no_choice_manifest[
        "operational_failures"
    ]
    assert no_choice_manifest.self_world["WORLD"]["available_interfaces"][
        "entries"
    ] == []
