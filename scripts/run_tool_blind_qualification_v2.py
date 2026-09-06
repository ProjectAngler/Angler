#!/usr/bin/env python3
"""Three-arm cloned-state witness for tool-blind autonomous target formation.

The source is an already sealed Jenny state, never the live state directory.
Arm A uses intact formation evidence and the authored-artifact registration.
Arm B removes A-cited evidence only while the target is formed. Arm C replays
A's exact frozen target with no target-formation backend call and removes only
the authored-artifact registration. An empty A target is honest evidence, but
can never set ``passed`` true.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Mapping, Sequence

from angler.runtime.authored_artifact import (
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AUTHORED_ARTIFACT_SOURCE_KIND,
    AUTHORED_ARTIFACT_SOURCE_REF,
    AuthoredArtifactAction,
)
from angler.runtime.jenny2_qualification import (
    PreparedQualificationClone,
    prepare_qualification_clone,
    state_tree_manifest,
)
from angler.runtime.jenny2_runtime import (
    SGLangLoRABinding,
    assemble_qwen38_autonomous_jenny2_with_cognee,
)
from angler.runtime.temporal_v2 import TrustedClock
import angler.runtime.authored_artifact as authored_artifact_module
import angler.runtime.frozen_cognitive_models as frozen_cognitive_models_module
import angler.runtime.higher_level_autonomy_adapter as autonomy_adapter_module
import angler.runtime.jenny2_qualification as qualification_module
import angler.runtime.jenny2_runtime as jenny2_runtime_module
import angler.runtime.persistent_autonomy as persistent_autonomy_module
import angler.runtime.temporal_v2 as temporal_v2_module


SCHEMA = "jenny2.tool-blind-authored-artifact-qualification.v2"
PREDECESSOR_SCHEMA = "jenny2.authored-artifact-cloned-state-qualification.v1"
PRESERVED_CONSUMED_IDENTITIES = {
    "r1": {
        "qualification_id": "jenny2-authored-artifact-r1",
        "disposition": "INVALID",
        "schema": PREDECESSOR_SCHEMA,
    },
    "r2": {
        "qualification_id": "jenny2-authored-artifact-r2",
        "disposition": "HONEST_QUIESCENT",
        "schema": PREDECESSOR_SCHEMA,
    },
    "r3": {
        "qualification_id": "jenny2-authored-artifact-r3",
        "disposition": "HONEST_QUIESCENT",
        "schema": PREDECESSOR_SCHEMA,
    },
}
_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}")
_CREDIT_FIELDS = (
    "affordance_utility",
    "memory_utility",
    "affordance_outcome_profiles",
    "capability_evidence",
    "capability_use_evidence",
)
_ARM_NAMES = (
    "a_intact_artifact",
    "b_cited_evidence_lesion_artifact",
    "c_frozen_target_registration_removed",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_ref(value: object) -> str:
    return "sha256:" + sha256(_canonical(value)).hexdigest()


def _json_copy(value: object):
    return json.loads(_canonical(value))


def _preserved_identity_gate(qualification_id: str) -> bool:
    if _ID.fullmatch(qualification_id) is None:
        raise ValueError("qualification_id must be a bounded lowercase identity")
    consumed = {
        item["qualification_id"] for item in PRESERVED_CONSUMED_IDENTITIES.values()
    }
    if qualification_id in consumed:
        raise ValueError("qualification_id collides with a consumed r1/r2/r3 identity")
    return True


def _state_payload(runtime: object) -> dict[str, object]:
    value = json.loads(runtime.supervisor.state_bytes())  # type: ignore[attr-defined]
    if type(value) is not dict:
        raise RuntimeError("canonical Jenny state is not an object")
    return value


def _credits(state: Mapping[str, object]) -> dict[str, object]:
    return {name: state.get(name) for name in _CREDIT_FIELDS}


def _status(runtime: object) -> dict[str, object]:
    value = asdict(runtime.status())  # type: ignore[attr-defined]
    required = {
        "external_effects_enabled",
        "last_event_ref",
        "moving_origin_ordinal",
        "pending_operation",
        "pending_projections",
        "scheduler_enabled",
        "state_ref",
    }
    if not required.issubset(value):
        raise RuntimeError("runtime status schema differs")
    result = {name: value[name] for name in sorted(required)}
    if result["external_effects_enabled"] is not False:
        raise RuntimeError("qualification runtime exposed external effects")
    if result["pending_operation"] is not False or result["pending_projections"] != 0:
        raise RuntimeError("qualification runtime did not begin settled")
    return result


def _db_snapshot(database: Path) -> dict[str, object]:
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        episodes = tuple(
            connection.execute(
                "SELECT episode_ref,trigger_ref,ordinal,event_ref FROM episodes "
                "ORDER BY ordinal"
            ).fetchall()
        )
        projections = tuple(
            connection.execute(
                "SELECT episode_ref,event_ref,ordinal,backend_ref "
                "FROM projection_outbox ORDER BY ordinal"
            ).fetchall()
        )
    return {"episodes": episodes, "projections": projections}


def _sqlite_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"sqlite_blob_hex": value.hex()}
    if isinstance(value, tuple):
        return [_sqlite_value(item) for item in value]
    return value


def _sqlite_logical_snapshot(
    database: Path, *, excluded_tables: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Return an order-stable logical snapshot for exact mutation checks."""

    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        names = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if row[0] not in excluded_tables
        )
        result: dict[str, object] = {}
        for name in names:
            quoted = '"' + str(name).replace('"', '""') + '"'
            columns = tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")
            )
            rows = tuple(connection.execute(f"SELECT * FROM {quoted} ORDER BY rowid"))
            result[str(name)] = {
                "columns": list(columns),
                "rows": [_sqlite_value(tuple(row)) for row in rows],
            }
    return result


def _rearm_autonomy_wake(database: Path) -> dict[str, object]:
    """Remove only the clone's mechanical wake-deduplication checkpoint."""

    excluded = frozenset(("autonomy_wake_checkpoint",))
    before_other = _sqlite_logical_snapshot(database, excluded_tables=excluded)
    with closing(sqlite3.connect(database)) as connection:
        checkpoint_before = tuple(
            connection.execute(
                "SELECT state_ref,last_event_ref,checkpoint_ref "
                "FROM autonomy_wake_checkpoint ORDER BY singleton"
            )
        )
        if len(checkpoint_before) > 1:
            raise RuntimeError("wake checkpoint singleton contains multiple rows")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM autonomy_wake_checkpoint WHERE singleton=1")
        connection.commit()
        checkpoint_after = tuple(
            connection.execute(
                "SELECT state_ref,last_event_ref,checkpoint_ref "
                "FROM autonomy_wake_checkpoint ORDER BY singleton"
            )
        )
    after_other = _sqlite_logical_snapshot(database, excluded_tables=excluded)
    if after_other != before_other:
        raise RuntimeError("re-arming the clone changed non-checkpoint logical state")
    if checkpoint_after:
        raise RuntimeError("wake checkpoint remained after re-arm")
    return {
        "removed_rows": len(checkpoint_before),
        "checkpoint_before_ref": _content_ref(_sqlite_value(checkpoint_before)),
        "checkpoint_after_empty": True,
        "noncheckpoint_snapshot_ref": _content_ref(before_other),
        "noncheckpoint_unchanged": True,
    }


@dataclass(frozen=True)
class FixedTemporalInput:
    trusted_utc: str
    monotonic_ns: int
    timezone_name: str = "America/New_York"
    uncertainty_ms: float = 1.0
    jump_tolerance_ms: float = 1_000.0

    @classmethod
    def capture(cls) -> "FixedTemporalInput":
        wall = datetime.now(timezone.utc)
        return cls(
            trusted_utc=wall.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            monotonic_ns=time.monotonic_ns(),
        )

    def clock(self) -> TrustedClock:
        wall = datetime.fromisoformat(self.trusted_utc[:-1] + "+00:00")
        monotonic = self.monotonic_ns
        return TrustedClock(
            timezone_name=self.timezone_name,
            uncertainty_ms=self.uncertainty_ms,
            jump_tolerance_ms=self.jump_tolerance_ms,
            wall_clock=lambda: wall,
            monotonic_clock=lambda: monotonic,
        )

    def payload(self) -> dict[str, object]:
        sample = self.clock().sample(-1)
        return {
            **asdict(self),
            "clock_anchor_ref": sample.clock_anchor_ref,
            "fixed_wall_and_monotonic_for_all_reads": True,
        }


def _assemble_arm(
    prepared: PreparedQualificationClone,
    binding: SGLangLoRABinding,
    binding_path: Path,
    served_model: str,
    library_root: Path,
    *,
    include_authored_artifact: bool,
    clock: TrustedClock,
):
    kwargs = dict(
        genesis_created_at_utc=prepared.genesis_created_at_utc,
        cognee_scope=prepared.primary_scope,
        capability_cognee_scope=prepared.capability_scope,
        runtime_ref=binding.runtime_ref,
        endpoint=binding.endpoint,
        served_model=served_model,
        model_binding_path=binding_path,
        selector_qualification_ref=binding.selector_qualification_ref,
        enable_native_openclaw=False,
        library_root=library_root,
        clock=clock,
    )
    if include_authored_artifact:
        return assemble_qwen38_autonomous_jenny2_with_cognee(
            prepared.root / "jenny2.sqlite3", **kwargs
        )

    original = jenny2_runtime_module.assemble_jenny2_runtime

    def assemble_without_authored(*args, **inner_kwargs):
        bindings = tuple(inner_kwargs.get("internal_affordance_bindings", ()))
        retained = tuple(
            item
            for item in bindings
            if item.affordance.affordance_id != AUTHORED_ARTIFACT_AFFORDANCE_ID
        )
        if len(bindings) - len(retained) != 1:
            raise RuntimeError("removal control did not remove exactly one registration")
        inner_kwargs["internal_affordance_bindings"] = retained
        return original(*args, **inner_kwargs)

    jenny2_runtime_module.assemble_jenny2_runtime = assemble_without_authored
    try:
        return assemble_qwen38_autonomous_jenny2_with_cognee(
            prepared.root / "jenny2.sqlite3", **kwargs
        )
    finally:
        jenny2_runtime_module.assemble_jenny2_runtime = original


def _callable_identity(value: object) -> dict[str, str]:
    owner = getattr(value, "__self__", None)
    target = owner if owner is not None else value
    target_type = target if inspect.isclass(target) else type(target)
    return {
        "module": target_type.__module__,
        "qualname": target_type.__qualname__,
    }


def _registrations(runtime: object) -> tuple[dict[str, object], ...]:
    registry = runtime.supervisor.affordances  # type: ignore[attr-defined]
    result: list[dict[str, object]] = []
    for item in registry.definitions():
        executor = None
        if item.disposition == "ACT" and registry.execution_mode(item.affordance_id) == "SYNC":
            executor = _callable_identity(registry.executor(item.affordance_id))
        result.append(
            {
                "affordance": asdict(item),
                "execution_mode": registry.execution_mode(item.affordance_id),
                "observable_source_ref": registry.observable_source_ref(item.affordance_id),
                "observable_source_kind": registry.observable_source_kind(item.affordance_id),
                "consequence_mapper_ref": registry.consequence_mapper_ref(item.affordance_id),
                "executor": executor,
            }
        )
    return tuple(result)


def _memory_payload(item: object) -> dict[str, object]:
    if is_dataclass(item):
        value = asdict(item)
    elif isinstance(item, Mapping):
        value = dict(item)
    else:
        raise TypeError("formation memory must be a dataclass or mapping")
    if type(value) is not dict:
        raise TypeError("formation memory payload must be an object")
    return value


def _formation_input_payload(
    *,
    cognitive_state: Mapping[str, object],
    temporal: Mapping[str, object],
    memories: Sequence[object],
    evidence_catalog: Mapping[str, str],
) -> dict[str, object]:
    return {
        "cognitive_state": dict(cognitive_state),
        "evidence_catalog": dict(evidence_catalog),
        "retrieved_memories": [_memory_payload(item) for item in memories],
        "temporal": dict(temporal),
    }


def _canonical_evidence_catalog(
    *,
    cognitive_state: Mapping[str, object],
    memories: Sequence[object],
    canonical_state_ref: str,
    temporal_sample_ref: str,
) -> dict[str, str]:
    catalog = {
        "canonical-state": canonical_state_ref,
        "temporal-now": temporal_sample_ref,
    }
    for index, item in enumerate(memories, start=1):
        reference = _memory_payload(item).get("record_ref")
        if type(reference) is not str or _SHA256_REF.fullmatch(reference) is None:
            raise ValueError("formation memory record_ref is malformed")
        catalog[f"memory-{index}"] = reference
    for field, value in cognitive_state.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        catalog["state-" + field.replace("_", "-")] = _content_ref(
            {
                "canonical_state_ref": canonical_state_ref,
                "field": field,
                "value": value,
            }
        )
    return catalog


@dataclass(frozen=True)
class LesionedFormationInputs:
    cognitive_state: dict[str, object]
    temporal: dict[str, object]
    memories: tuple[object, ...]
    evidence_catalog: dict[str, str]
    receipt: dict[str, object]


def _lesion_formation_inputs(
    *,
    cognitive_state: Mapping[str, object],
    temporal: Mapping[str, object],
    memories: Sequence[object],
    evidence_catalog: Mapping[str, str],
    cited_keys: Sequence[str],
) -> LesionedFormationInputs:
    """Remove A-cited grounding while retaining fixed state/time anchors."""

    original_state = dict(cognitive_state)
    original_temporal = dict(temporal)
    original_memories = tuple(memories)
    original_catalog = dict(evidence_catalog)
    keys = tuple(cited_keys)
    if len(keys) != len(set(keys)) or any(key not in original_catalog for key in keys):
        raise ValueError("cited lesion keys must be unique members of A's catalog")
    grounding_keys = tuple(
        key for key in keys if key.startswith(("state-", "memory-"))
    )
    retained_anchor_keys = tuple(key for key in keys if key not in grounding_keys)
    if any(key not in {"canonical-state", "temporal-now"} for key in retained_anchor_keys):
        raise ValueError("A cited an unsupported non-grounding evidence key")
    canonical_state_ref = original_catalog.get("canonical-state")
    temporal_sample_ref = original_catalog.get("temporal-now")
    if (
        type(canonical_state_ref) is not str
        or _SHA256_REF.fullmatch(canonical_state_ref) is None
        or type(temporal_sample_ref) is not str
        or _SHA256_REF.fullmatch(temporal_sample_ref) is None
    ):
        raise ValueError("formation evidence anchors are malformed")
    expected_original_catalog = _canonical_evidence_catalog(
        cognitive_state=original_state,
        memories=original_memories,
        canonical_state_ref=canonical_state_ref,
        temporal_sample_ref=temporal_sample_ref,
    )
    if original_catalog != expected_original_catalog:
        raise ValueError("A's formation evidence catalog is not canonical")

    state_key_to_field = {
        "state-" + field.replace("_", "-"): field for field in original_state
    }
    memory_key_to_index = {
        f"memory-{index}": index - 1
        for index in range(1, len(original_memories) + 1)
    }
    removed_state_fields = tuple(
        sorted(
            state_key_to_field[key] for key in grounding_keys
            if key in state_key_to_field
        )
    )
    removed_memory_indices = tuple(
        sorted(
            memory_key_to_index[key] for key in grounding_keys
            if key in memory_key_to_index
        )
    )
    forwarded_state = {
        field: value
        for field, value in original_state.items()
        if field not in set(removed_state_fields)
    }
    forwarded_memories = tuple(
        item
        for index, item in enumerate(original_memories)
        if index not in set(removed_memory_indices)
    )
    forwarded_catalog = _canonical_evidence_catalog(
        cognitive_state=forwarded_state,
        memories=forwarded_memories,
        canonical_state_ref=canonical_state_ref,
        temporal_sample_ref=temporal_sample_ref,
    )

    source_payload = _formation_input_payload(
        cognitive_state=original_state,
        temporal=original_temporal,
        memories=original_memories,
        evidence_catalog=original_catalog,
    )
    forwarded_payload = _formation_input_payload(
        cognitive_state=forwarded_state,
        temporal=original_temporal,
        memories=forwarded_memories,
        evidence_catalog=forwarded_catalog,
    )
    removed_memory_refs = tuple(
        _memory_payload(original_memories[index]).get("record_ref")
        for index in removed_memory_indices
    )
    retained_memory_indices = tuple(
        index
        for index in range(len(original_memories))
        if index not in set(removed_memory_indices)
    )
    memory_key_reindex = {
        f"memory-{old_index + 1}": f"memory-{new_index}"
        for new_index, old_index in enumerate(retained_memory_indices, start=1)
    }
    receipt = {
        "cited_keys": list(keys),
        "lesioned_grounding_keys": list(grounding_keys),
        "retained_anchor_keys": list(retained_anchor_keys),
        "removed_catalog_entries": {
            key: original_catalog[key] for key in grounding_keys
        },
        "removed_evidence_refs": [original_catalog[key] for key in grounding_keys],
        "removed_state_fields": list(removed_state_fields),
        "removed_memory_indices": list(removed_memory_indices),
        "removed_memory_refs": list(removed_memory_refs),
        "memory_key_reindex": memory_key_reindex,
        "source_model_input_ref": _content_ref(source_payload),
        "forwarded_model_input_ref": _content_ref(forwarded_payload),
        "source_formation_input_ref": None,
        "forwarded_formation_input_ref": None,
        "temporal_unchanged": original_temporal == dict(temporal),
        "anchors_unchanged": all(
            forwarded_catalog[key] == original_catalog[key]
            for key in ("canonical-state", "temporal-now")
        ),
        "catalog_exact": forwarded_catalog
        == _canonical_evidence_catalog(
            cognitive_state=forwarded_state,
            memories=forwarded_memories,
            canonical_state_ref=canonical_state_ref,
            temporal_sample_ref=temporal_sample_ref,
        ),
        "forwarded_catalog_ref": _content_ref(forwarded_catalog),
        "state_payload_exact": forwarded_state
        == {
            field: value
            for field, value in original_state.items()
            if field not in set(removed_state_fields)
        },
        "memory_payload_exact": [
            _memory_payload(item) for item in forwarded_memories
        ]
        == [
            _memory_payload(item)
            for index, item in enumerate(original_memories)
            if index not in set(removed_memory_indices)
        ],
    }
    return LesionedFormationInputs(
        cognitive_state=forwarded_state,
        temporal=original_temporal,
        memories=forwarded_memories,
        evidence_catalog=forwarded_catalog,
        receipt=receipt,
    )


class FormationIntervention:
    """Instrument formation without altering canonical state or operation inputs."""

    def __init__(
        self,
        mode: str,
        *,
        cited_keys: Sequence[str] = (),
        replay_formation: Mapping[str, object] | None = None,
    ) -> None:
        if mode not in {"INTACT", "CITED_EVIDENCE_LESION", "FROZEN_REPLAY"}:
            raise ValueError("unknown formation intervention mode")
        if mode == "FROZEN_REPLAY" and replay_formation is None:
            raise ValueError("frozen replay requires A's formation")
        if mode != "FROZEN_REPLAY" and replay_formation is not None:
            raise ValueError("only frozen replay may receive A's formation")
        self.mode = mode
        self.cited_keys = tuple(cited_keys)
        self.replay_formation = (
            None if replay_formation is None else _json_copy(dict(replay_formation))
        )
        self.wrapper_calls = 0
        self.original_method_calls = 0
        self.backend_calls = 0
        self.proposal_phase_backend_calls = 0
        self.source_input_ref: str | None = None
        self.forwarded_input_ref: str | None = None
        self.source_evidence_catalog: dict[str, str] | None = None
        self.forwarded_evidence_catalog: dict[str, str] | None = None
        self.lesion_receipt: dict[str, object] | None = None
        self.record_rebound = False
        self.formation_handoff_rebound = False
        self.replay_exact = False
        self._model = None
        self._cycle = None
        self._original_form = None
        self._original_propose = None
        self._source_input_payload: dict[str, object] | None = None
        self._forwarded_input_payload: dict[str, object] | None = None

    def _invoke_original(self, **kwargs):
        if self._original_form is None:
            raise RuntimeError("formation intervention is not installed")
        self.original_method_calls += 1
        backend = getattr(self._model, "backend", None)
        generate = getattr(backend, "generate", None)
        if not callable(generate):
            raise RuntimeError("target-forming model exposes no countable backend")

        def counted_generate(*args, **inner_kwargs):
            self.backend_calls += 1
            return generate(*args, **inner_kwargs)

        setattr(backend, "generate", counted_generate)
        try:
            return self._original_form(**kwargs)
        finally:
            setattr(backend, "generate", generate)

    def _form(self, **kwargs):
        self.wrapper_calls += 1
        if self.wrapper_calls != 1:
            raise RuntimeError("one qualification arm attempted target formation twice")
        required = {"cognitive_state", "temporal", "memories", "evidence_catalog"}
        if set(kwargs) != required:
            raise RuntimeError("target formation call schema differs")
        source_payload = _formation_input_payload(**kwargs)
        self._source_input_payload = _json_copy(source_payload)
        self.source_input_ref = _content_ref(source_payload)
        self.source_evidence_catalog = dict(kwargs["evidence_catalog"])

        if self.mode == "FROZEN_REPLAY":
            self.forwarded_input_ref = self.source_input_ref
            self._forwarded_input_payload = _json_copy(source_payload)
            self.forwarded_evidence_catalog = dict(kwargs["evidence_catalog"])
            assert self.replay_formation is not None
            proposal = self.replay_formation.get("proposal")
            if type(proposal) is not dict:
                raise RuntimeError("A's frozen formation contains no proposal")
            return _json_copy(proposal)

        forwarded = kwargs
        if self.mode == "CITED_EVIDENCE_LESION":
            lesion = _lesion_formation_inputs(
                cognitive_state=kwargs["cognitive_state"],
                temporal=kwargs["temporal"],
                memories=kwargs["memories"],
                evidence_catalog=kwargs["evidence_catalog"],
                cited_keys=self.cited_keys,
            )
            self.lesion_receipt = lesion.receipt
            forwarded = {
                "cognitive_state": lesion.cognitive_state,
                "temporal": lesion.temporal,
                "memories": lesion.memories,
                "evidence_catalog": lesion.evidence_catalog,
            }
        forwarded_payload = _formation_input_payload(**forwarded)
        self.forwarded_input_ref = _content_ref(forwarded_payload)
        self._forwarded_input_payload = _json_copy(forwarded_payload)
        self.forwarded_evidence_catalog = dict(forwarded["evidence_catalog"])
        return self._invoke_original(**forwarded)

    @staticmethod
    def _retained_core_input(formation_input: Mapping[str, object]) -> dict[str, object]:
        required = {
            "cognitive_state",
            "evidence_catalog",
            "retrieved_memories",
            "temporal",
        }
        if not required.issubset(formation_input):
            raise RuntimeError("retained formation input lacks the model-call payload")
        return {key: formation_input[key] for key in sorted(required)}

    def _bind_intact_formation(self, formation: Mapping[str, object]) -> None:
        if self._source_input_payload is None or self._forwarded_input_payload is None:
            raise RuntimeError("formation input was not captured")
        retained = formation.get("formation_input")
        if retained is None:
            return
        if type(retained) is not dict:
            raise RuntimeError("retained formation input is malformed")
        core = self._retained_core_input(retained)
        if core != self._source_input_payload or core != self._forwarded_input_payload:
            raise RuntimeError("retained intact formation input differs from the call seam")
        retained_ref = _content_ref(retained)
        if formation.get("formation_input_ref") != retained_ref:
            raise RuntimeError("retained intact formation input identity differs")
        self.source_input_ref = retained_ref
        self.forwarded_input_ref = retained_ref

    def _rebind_lesioned_formation(self) -> None:
        if (
            self._cycle is None
            or self.forwarded_input_ref is None
            or self._source_input_payload is None
            or self._forwarded_input_payload is None
        ):
            raise RuntimeError("lesion formation was not captured")
        formation = getattr(self._cycle, "last_autonomous_formation", None)
        if type(formation) is not dict:
            raise RuntimeError("cycle exposes no completed target formation")
        proposal = formation.get("proposal")
        if type(proposal) is not dict or self.forwarded_evidence_catalog is None:
            raise RuntimeError("lesioned target proposal is malformed")
        keys = proposal.get("evidence_keys")
        if (
            type(keys) is not list
            or any(key not in self.forwarded_evidence_catalog for key in keys)
        ):
            raise RuntimeError("lesioned target cites removed evidence")
        retained = formation.get("formation_input")
        if retained is not None:
            if type(retained) is not dict:
                raise RuntimeError("retained formation input is malformed")
            if self._retained_core_input(retained) != self._source_input_payload:
                raise RuntimeError("retained source formation input differs from the call seam")
            source_formation_input_ref = _content_ref(retained)
            if formation.get("formation_input_ref") != source_formation_input_ref:
                raise RuntimeError("retained source formation input identity differs")
            forwarded_retained = _json_copy(retained)
            forwarded_retained.update(self._forwarded_input_payload)
            forwarded_formation_input_ref = _content_ref(forwarded_retained)
            formation["formation_input"] = forwarded_retained
            formation["formation_input_ref"] = forwarded_formation_input_ref
            self.source_input_ref = source_formation_input_ref
            self.forwarded_input_ref = forwarded_formation_input_ref
        else:
            formation["formation_input_ref"] = self.forwarded_input_ref
        if self.lesion_receipt is None:
            raise RuntimeError("cited lesion receipt is missing")
        self.lesion_receipt["source_formation_input_ref"] = self.source_input_ref
        self.lesion_receipt[
            "forwarded_formation_input_ref"
        ] = self.forwarded_input_ref
        formation["formation_evidence_refs"] = [
            self.forwarded_evidence_catalog[key] for key in keys
        ]
        unhashed = dict(formation)
        unhashed.pop("formation_ref", None)
        formation["formation_ref"] = _content_ref(unhashed)
        setattr(self._cycle, "_last_autonomous_formation", _json_copy(formation))
        pending = getattr(self._cycle, "_pending_autonomous_initiative", None)
        self.formation_handoff_rebound = pending is None
        if pending is not None:
            if type(pending) is not dict or pending.get("proposal") != proposal:
                raise RuntimeError("pending initiative differs from lesioned formation")
            if pending.get("evidence_catalog") != self.source_evidence_catalog:
                raise RuntimeError("source formation handoff catalog differs")
            rebound = _json_copy(pending)
            rebound["formation"] = _json_copy(formation)
            rebound["evidence_catalog"] = dict(self.forwarded_evidence_catalog)
            setattr(self._cycle, "_pending_autonomous_initiative", rebound)
            self.formation_handoff_rebound = (
                rebound.get("evidence_catalog") == self.forwarded_evidence_catalog
            )
        self.record_rebound = True

    def _after_propose(self) -> None:
        if self._cycle is None:
            raise RuntimeError("formation intervention is not installed")
        if self.mode == "CITED_EVIDENCE_LESION":
            self._rebind_lesioned_formation()
        else:
            actual = getattr(self._cycle, "last_autonomous_formation", None)
            if type(actual) is not dict:
                raise RuntimeError("cycle exposes no completed target formation")
            self._bind_intact_formation(actual)
        if self.mode == "FROZEN_REPLAY":
            self.replay_exact = actual == self.replay_formation
            if not self.replay_exact:
                raise RuntimeError("frozen A target did not replay byte-exactly")

    def install(self, runtime: object) -> None:
        cycle = runtime.supervisor.cycle  # type: ignore[attr-defined]
        model = getattr(cycle, "experience_model", None)
        form = getattr(model, "form_autonomous_target", None)
        propose = getattr(cycle, "propose_autonomous_request", None)
        if not callable(form) or not callable(propose):
            raise RuntimeError("runtime lacks the tool-blind target-formation boundary")
        if not hasattr(cycle, "last_autonomous_formation"):
            raise RuntimeError("runtime lacks immutable formation diagnostics")
        self._cycle = cycle
        self._model = model
        self._original_form = form
        self._original_propose = propose
        setattr(model, "form_autonomous_target", self._form)

        def proposed(**kwargs):
            assert self._original_propose is not None
            backend = getattr(self._model, "backend", None)
            generate = getattr(backend, "generate", None)
            if not callable(generate):
                raise RuntimeError("target-forming model exposes no countable backend")

            def counted_phase_generate(*args, **inner_kwargs):
                self.proposal_phase_backend_calls += 1
                return generate(*args, **inner_kwargs)

            setattr(backend, "generate", counted_phase_generate)
            try:
                result = self._original_propose(**kwargs)
            finally:
                setattr(backend, "generate", generate)
            self._after_propose()
            return result

        setattr(cycle, "propose_autonomous_request", proposed)

    def uninstall(self) -> None:
        if self._model is not None and self._original_form is not None:
            setattr(self._model, "form_autonomous_target", self._original_form)
        if self._cycle is not None and self._original_propose is not None:
            setattr(self._cycle, "propose_autonomous_request", self._original_propose)

    def payload(self, formation: Mapping[str, object] | None) -> dict[str, object]:
        return {
            "mode": self.mode,
            "wrapper_calls": self.wrapper_calls,
            "original_method_calls": self.original_method_calls,
            "formation_backend_calls": self.backend_calls,
            "proposal_phase_backend_calls": self.proposal_phase_backend_calls,
            "source_input_ref": self.source_input_ref,
            "forwarded_input_ref": self.forwarded_input_ref,
            "source_evidence_catalog": self.source_evidence_catalog,
            "forwarded_evidence_catalog": self.forwarded_evidence_catalog,
            "cited_lesion_keys": list(self.cited_keys),
            "lesion_receipt": self.lesion_receipt,
            "record_rebound": self.record_rebound,
            "formation_handoff_rebound": self.formation_handoff_rebound,
            "replay_exact": self.replay_exact,
            "formation_ref": None if formation is None else formation.get("formation_ref"),
        }


def _episode(runtime: object, episode_ref: str) -> dict[str, object]:
    item = runtime.supervisor.episode_item(episode_ref)  # type: ignore[attr-defined]
    value = json.loads(item.payload_json)
    if type(value) is not dict:
        raise RuntimeError("committed episode is malformed")
    return value


@dataclass(frozen=True)
class ArmCapture:
    name: str
    result: dict[str, object]
    before_status: dict[str, object]
    after_status: dict[str, object]
    before_state: bytes
    after_state: bytes
    before_db: dict[str, object]
    after_db: dict[str, object]
    before_credit: dict[str, object]
    after_credit: dict[str, object]
    before_artifacts: tuple[object, ...]
    after_artifacts: tuple[object, ...]
    episode: dict[str, object] | None
    registrations: tuple[dict[str, object], ...]
    formation: dict[str, object] | None
    formation_trace: dict[str, object]


def _capture_arm(
    runtime: object,
    *,
    name: str,
    trigger_ref: str,
    intervention: FormationIntervention,
) -> ArmCapture:
    database = Path(runtime.supervisor.path)  # type: ignore[attr-defined]
    before_status = _status(runtime)
    before_state = runtime.supervisor.state_bytes()  # type: ignore[attr-defined]
    before_payload = _state_payload(runtime)
    before_db = _db_snapshot(database)
    registrations = _registrations(runtime)
    if any(item["affordance"]["external_effect"] is not False for item in registrations):
        raise RuntimeError("qualification catalog contains an external effect")
    result_object = runtime.supervisor.scheduler_tick(trigger_ref)  # type: ignore[attr-defined]
    result = asdict(result_object)
    if result_object.status == "COMMITTED":
        runtime.drain_pending_projections()  # type: ignore[attr-defined]
    after_status = _status(runtime)
    after_state = runtime.supervisor.state_bytes()  # type: ignore[attr-defined]
    after_payload = _state_payload(runtime)
    after_db = _db_snapshot(database)
    episode = (
        None
        if result_object.episode_ref is None
        else _episode(runtime, result_object.episode_ref)
    )
    formation = getattr(  # type: ignore[attr-defined]
        runtime.supervisor.cycle, "last_autonomous_formation", None
    )
    if formation is not None and type(formation) is not dict:
        raise RuntimeError("runtime formation diagnostic is malformed")
    return ArmCapture(
        name=name,
        result=result,
        before_status=before_status,
        after_status=after_status,
        before_state=before_state,
        after_state=after_state,
        before_db=before_db,
        after_db=after_db,
        before_credit=_credits(before_payload),
        after_credit=_credits(after_payload),
        before_artifacts=tuple(before_payload.get("authored_artifacts", ())),
        after_artifacts=tuple(after_payload.get("authored_artifacts", ())),
        episode=episode,
        registrations=registrations,
        formation=None if formation is None else _json_copy(formation),
        formation_trace=intervention.payload(formation),
    )


def _artifact_gates(arm: ArmCapture) -> tuple[dict[str, bool], dict[str, object] | None]:
    result = arm.result
    selected = result.get("selected_affordance_id") == AUTHORED_ARTIFACT_AFFORDANCE_ID
    artifact: dict[str, object] | None = None
    identity_valid = False
    formation_evidence_linked = False
    if selected and len(arm.after_artifacts) == len(arm.before_artifacts) + 1:
        candidate = arm.after_artifacts[-1]
        if type(candidate) is dict and type(candidate.get("artifact")) is dict:
            action = AuthoredArtifactAction.from_json(
                _canonical(candidate["artifact"]).decode("utf-8")
            )
            identity_valid = candidate.get("artifact_ref") == action.artifact_ref
            artifact = {
                "artifact": action.canonical_payload(),
                "artifact_ref": action.artifact_ref,
                "entry_ref": candidate.get("entry_ref"),
                "epistemic_status": candidate.get("epistemic_status"),
            }
            formation_refs = (
                None
                if arm.formation is None
                else arm.formation.get("formation_evidence_refs")
            )
            formation_evidence_linked = bool(
                type(formation_refs) is list
                and formation_refs
                and set(formation_refs).issubset(action.evidence_refs)
            )
    episode = arm.episode or {}
    receipt = episode.get("receipt")
    observable = receipt.get("observable_consequence") if type(receipt) is dict else None
    projections_before = arm.before_db["projections"]
    projections_after = arm.after_db["projections"]
    episode_ref = result.get("episode_ref")
    new_projection = (
        projections_after[-1]
        if len(projections_after) == len(projections_before) + 1
        else None
    )
    return {
        "selected_authored_artifact": selected,
        "committed": result.get("status") == "COMMITTED",
        "one_artifact_appended": len(arm.after_artifacts)
        == len(arm.before_artifacts) + 1,
        "content_identity_valid": identity_valid,
        "artifact_cites_target_formation_evidence": formation_evidence_linked,
        "one_episode_committed": len(arm.after_db["episodes"])
        == len(arm.before_db["episodes"]) + 1,
        "moving_origin_advanced_once": arm.after_status["moving_origin_ordinal"]
        == arm.before_status["moving_origin_ordinal"] + 1,
        "projection_acknowledged_once": bool(
            new_projection is not None
            and new_projection[0] == episode_ref
            and type(new_projection[3]) is str
            and _SHA256_REF.fullmatch(new_projection[3]) is not None
        ),
        "receipt_has_no_scalar_consequence": bool(
            type(receipt) is dict and receipt.get("consequence") == []
        ),
        "receipt_is_source_bound": bool(
            type(observable) is dict
            and observable.get("source_kind") == AUTHORED_ARTIFACT_SOURCE_KIND
            and observable.get("source_ref") == AUTHORED_ARTIFACT_SOURCE_REF
            and artifact is not None
            and observable.get("artifact_refs") == [artifact["artifact_ref"]]
        ),
        "credit_unchanged": arm.after_credit == arm.before_credit,
        "settled": arm.after_status["pending_operation"] is False
        and arm.after_status["pending_projections"] == 0,
        "external_effects_disabled": arm.after_status["external_effects_enabled"] is False,
    }, artifact


def _quiescent_gates(arm: ArmCapture) -> dict[str, bool]:
    return {
        "honest_quiescent": arm.result.get("status") == "QUIESCENT",
        "canonical_state_unchanged": arm.after_state == arm.before_state,
        "moving_origin_unchanged": arm.after_status["moving_origin_ordinal"]
        == arm.before_status["moving_origin_ordinal"],
        "episode_history_unchanged": arm.after_db["episodes"] == arm.before_db["episodes"],
        "projection_history_unchanged": arm.after_db["projections"]
        == arm.before_db["projections"],
        "artifact_history_unchanged": arm.after_artifacts == arm.before_artifacts,
        "credit_unchanged": arm.after_credit == arm.before_credit,
    }


def _registration_map(arm: ArmCapture) -> dict[str, dict[str, object]]:
    result = {
        item["affordance"]["affordance_id"]: item for item in arm.registrations
    }
    if len(result) != len(arm.registrations):
        raise RuntimeError("qualification registration manifest contains duplicates")
    return result


def _formation_request(arm: ArmCapture) -> str | None:
    formation = arm.formation
    proposal = None if formation is None else formation.get("proposal")
    request = proposal.get("internal_request") if type(proposal) is dict else None
    return request if type(request) is str else None


def _formation_integrity(arm: ArmCapture, *, expected_mode: str) -> dict[str, bool]:
    formation = arm.formation or {}
    trace = arm.formation_trace
    validator = getattr(
        autonomy_adapter_module, "_validated_autonomous_target_formation", None
    )
    runtime_validator_rejoined = False
    if callable(validator):
        try:
            validator(formation, require_tool_blind=True)
            runtime_validator_rejoined = True
        except (TypeError, ValueError):
            pass
    proposal = formation.get("proposal")
    keys = formation.get("formation_evidence_keys")
    catalog = trace.get("forwarded_evidence_catalog")
    expected_refs = (
        None
        if type(keys) is not list or type(catalog) is not dict
        else [catalog.get(key) for key in keys]
    )
    unhashed = dict(formation)
    formation_ref = unhashed.pop("formation_ref", None)
    retained_input = formation.get("formation_input")
    return {
        "formation_present": bool(formation),
        "runtime_validator_rejoined": runtime_validator_rejoined,
        "tool_blind_mode": formation.get("formation_mode")
        == "TOOL_BLIND_STATE_MEMORY_TIME",
        "state_ref_bound": formation.get("state_ref")
        == arm.before_status.get("state_ref"),
        "moving_origin_bound": formation.get("moving_origin_ordinal")
        == arm.before_status.get("moving_origin_ordinal"),
        "trace_mode_exact": trace.get("mode") == expected_mode,
        "one_wrapper_call": trace.get("wrapper_calls") == 1,
        "input_ref_bound": formation.get("formation_input_ref")
        == trace.get("forwarded_input_ref"),
        "retained_input_bound_when_present": "formation_input" not in formation
        or (
            type(retained_input) is dict
            and _content_ref(retained_input) == trace.get("forwarded_input_ref")
            and retained_input.get("evidence_catalog") == catalog
        ),
        "evidence_refs_bound": type(proposal) is dict
        and type(keys) is list
        and proposal.get("evidence_keys") == keys
        and formation.get("formation_evidence_refs") == expected_refs,
        "formation_ref_valid": type(formation_ref) is str
        and formation_ref == _content_ref(unhashed),
    }


def _classify(
    *,
    integrity_gates: Mapping[str, bool],
    causal_gates: Mapping[str, bool],
    a_request: str | None,
    a_quiescent: bool,
    a_selected_artifact: bool,
) -> tuple[str, bool]:
    if not integrity_gates or not all(integrity_gates.values()):
        return "INVALID", False
    if a_request is not None and not a_request.strip() and a_quiescent:
        return "HONEST_NULL", False
    if not a_selected_artifact:
        return "NO_ARTIFACT_SELECTED", False
    if all(causal_gates.values()):
        return "PASS", True
    return "NOT_SUPPORTED", False


def _zero_lesion_is_behaviorally_equivalent(
    *,
    lesioned_grounding_keys: Sequence[str],
    a_request: str | None,
    b_request: str | None,
    a_quiescent: bool,
    b_quiescent: bool,
    a_formation: Mapping[str, object] | None,
    b_formation: Mapping[str, object] | None,
) -> bool:
    """Compare the zero-lesion control without requiring sampled prose equality.

    When A cites no removable grounding, B receives the exact same target input.
    A second model sample may phrase its rationale differently, so byte equality
    would confuse ordinary generation variance with an integrity failure.  The
    control remains strict about inputs, identity, cited evidence, the null
    decision, and the externally observable quiescent result.
    """

    if lesioned_grounding_keys:
        return True
    if (
        a_request is None
        or b_request is None
        or a_request.strip()
        or b_request.strip()
        or not a_quiescent
        or not b_quiescent
        or type(a_formation) is not dict
        or type(b_formation) is not dict
    ):
        return False
    exact_fields = (
        "contract",
        "formation_input_ref",
        "moving_origin_ordinal",
        "state_ref",
        "target_contract_revision",
        "target_model_ref",
        "temporal_sample_ref",
    )
    if any(a_formation.get(field) != b_formation.get(field) for field in exact_fields):
        return False
    for formation in (a_formation, b_formation):
        proposal = formation.get("proposal")
        if (
            type(proposal) is not dict
            or proposal.get("internal_request") != ""
            or proposal.get("evidence_keys") != []
            or formation.get("formation_evidence_keys") != []
            or formation.get("formation_evidence_refs") != []
        ):
            return False
    return True


def _runtime_source_identities() -> dict[str, dict[str, str]]:
    modules = {
        "authored_artifact": authored_artifact_module,
        "frozen_cognitive_models": frozen_cognitive_models_module,
        "higher_level_autonomy_adapter": autonomy_adapter_module,
        "jenny2_qualification": qualification_module,
        "jenny2_runtime": jenny2_runtime_module,
        "persistent_autonomy": persistent_autonomy_module,
        "temporal_v2": temporal_v2_module,
    }
    result: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve(strict=True)
        result[name] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    return result


def _write_result(result_root: Path, payload: dict[str, object]) -> tuple[Path, str]:
    encoded = _canonical(payload)
    digest = sha256(encoded).hexdigest()
    path = result_root / f"{digest}.json"
    path.write_bytes(encoded)
    return path, digest


def run_qualification(
    *,
    live_state_root: Path,
    sealed_source_state: Path,
    binding_path: Path,
    library_root: Path,
    served_model: str | None,
    work_root: Path,
    result_root: Path,
    qualification_id: str,
    runtime_factory: Callable[..., object] = _assemble_arm,
    temporal_input: FixedTemporalInput | None = None,
) -> tuple[Path, str, str]:
    _preserved_identity_gate(qualification_id)
    for path, label in ((work_root, "work root"), (result_root, "result root")):
        if not path.is_absolute() or path.exists():
            raise ValueError(f"{label} must be a new canonical absolute path")
        if path.resolve(strict=False) != path:
            raise ValueError(f"{label} must be canonical")
    if (
        not live_state_root.is_absolute()
        or live_state_root.resolve(strict=True) != live_state_root
        or not live_state_root.is_dir()
    ):
        raise ValueError("live state root must be an exact canonical directory")
    if (
        not sealed_source_state.is_absolute()
        or sealed_source_state.resolve(strict=True) != sealed_source_state
        or not sealed_source_state.is_dir()
    ):
        raise ValueError("sealed source state must be an exact canonical directory")
    if (
        sealed_source_state == live_state_root
        or sealed_source_state in live_state_root.parents
        or live_state_root in sealed_source_state.parents
    ):
        raise ValueError("sealed source must be disjoint from the live state root")
    for output_root in (work_root, result_root):
        if any(
            output_root == protected
            or output_root in protected.parents
            or protected in output_root.parents
            for protected in (sealed_source_state, live_state_root)
        ):
            raise ValueError("live, sealed-source, work, and result roots must be disjoint")
    if (
        work_root == result_root
        or work_root in result_root.parents
        or result_root in work_root.parents
    ):
        raise ValueError("work and result roots must be disjoint")
    if (
        not binding_path.is_absolute()
        or binding_path.resolve(strict=True) != binding_path
        or not binding_path.is_file()
    ):
        raise ValueError("binding path must be an exact canonical file")
    if (
        not library_root.is_absolute()
        or library_root.resolve(strict=True) != library_root
        or not library_root.is_dir()
    ):
        raise ValueError("library root must be an exact canonical directory")
    if any(
        output_root == library_root
        or output_root in library_root.parents
        or library_root in output_root.parents
        for output_root in (work_root, result_root)
    ):
        raise ValueError("work and result roots must be disjoint from the library")

    # Qualification roots must be fresh, but their non-semantic container
    # directories need not have been provisioned by a separate command.
    work_root.mkdir(mode=0o700, parents=True)
    result_root.mkdir(mode=0o700, parents=True)
    started = time.perf_counter()
    source_before = state_tree_manifest(sealed_source_state)
    fixed_temporal = temporal_input or FixedTemporalInput.capture()

    def sealed_checker(source: Path) -> None:
        if source != sealed_source_state or state_tree_manifest(source) != source_before:
            raise RuntimeError("sealed source state changed during qualification")

    binding = SGLangLoRABinding.from_file(binding_path)
    if served_model is None:
        served_model = binding.served_model
    if binding.served_model != served_model or binding.selector_qualification_ref is None:
        raise ValueError("served model/binding qualification identity differs")

    prepared = {
        name: prepare_qualification_clone(
            sealed_source_state,
            work_root / name,
            offline_checker=sealed_checker,
        )
        for name in _ARM_NAMES
    }
    databases = {name: item.root / "jenny2.sqlite3" for name, item in prepared.items()}
    initial_bytes = {name: database.read_bytes() for name, database in databases.items()}
    if len(set(initial_bytes.values())) != 1:
        raise RuntimeError("qualification clone databases differ before intervention")
    rearm = {name: _rearm_autonomy_wake(database) for name, database in databases.items()}
    if len({_canonical(item) for item in rearm.values()}) != 1:
        raise RuntimeError("wake rearm mutation differs across arms")

    runtimes: dict[str, object] = {}
    captures: dict[str, ArmCapture] = {}
    trigger_ref = qualification_id + ":autonomous-wake"

    def execute_arm(
        name: str,
        *,
        include_authored_artifact: bool,
        intervention: FormationIntervention,
    ) -> ArmCapture:
        runtime = runtime_factory(
            prepared[name],
            binding,
            binding_path,
            served_model,
            library_root,
            include_authored_artifact=include_authored_artifact,
            clock=fixed_temporal.clock(),
        )
        runtimes[name] = runtime
        intervention.install(runtime)
        try:
            capture = _capture_arm(
                runtime,
                name=name,
                trigger_ref=trigger_ref,
                intervention=intervention,
            )
        finally:
            intervention.uninstall()
        runtime.close()  # type: ignore[attr-defined]
        del runtimes[name]
        captures[name] = capture
        return capture

    try:
        arm_a = execute_arm(
            _ARM_NAMES[0],
            include_authored_artifact=True,
            intervention=FormationIntervention("INTACT"),
        )
        if arm_a.formation is None:
            raise RuntimeError("arm A produced no target-formation envelope")
        cited_keys = arm_a.formation.get("formation_evidence_keys")
        if type(cited_keys) is not list or any(type(item) is not str for item in cited_keys):
            raise RuntimeError("arm A formation evidence keys are malformed")

        arm_b = execute_arm(
            _ARM_NAMES[1],
            include_authored_artifact=True,
            intervention=FormationIntervention(
                "CITED_EVIDENCE_LESION", cited_keys=tuple(cited_keys)
            ),
        )
        arm_c = execute_arm(
            _ARM_NAMES[2],
            include_authored_artifact=False,
            intervention=FormationIntervention(
                "FROZEN_REPLAY", replay_formation=arm_a.formation
            ),
        )

        restart_gates: dict[str, bool] = {}
        for name, include in zip(_ARM_NAMES, (True, True, False), strict=True):
            restarted = runtime_factory(
                prepared[name],
                binding,
                binding_path,
                served_model,
                library_root,
                include_authored_artifact=include,
                clock=fixed_temporal.clock(),
            )
            runtimes[name] = restarted
            capture = captures[name]
            restart_status = _status(restarted)
            restart_gates[name] = bool(
                restarted.supervisor.state_bytes()  # type: ignore[attr-defined]
                == capture.after_state
                and all(
                    restart_status[field] == capture.after_status[field]
                    for field in ("state_ref", "last_event_ref", "moving_origin_ordinal")
                )
            )
            restarted.close()  # type: ignore[attr-defined]
            del runtimes[name]

        a_map = _registration_map(arm_a)
        b_map = _registration_map(arm_b)
        c_map = _registration_map(arm_c)
        expected_c_map = {
            key: value for key, value in a_map.items()
            if key != AUTHORED_ARTIFACT_AFFORDANCE_ID
        }
        expected_c_registrations = tuple(
            item
            for item in arm_a.registrations
            if item["affordance"]["affordance_id"]
            != AUTHORED_ARTIFACT_AFFORDANCE_ID
        )
        a_request = _formation_request(arm_a)
        b_request = _formation_request(arm_b)
        a_formation_gates = _formation_integrity(arm_a, expected_mode="INTACT")
        b_formation_gates = _formation_integrity(
            arm_b, expected_mode="CITED_EVIDENCE_LESION"
        )
        c_formation_gates = _formation_integrity(
            arm_c, expected_mode="FROZEN_REPLAY"
        )
        a_artifact_gates, artifact = _artifact_gates(arm_a)
        a_quiescent_gates = _quiescent_gates(arm_a)
        b_quiescent_gates = _quiescent_gates(arm_b)
        lesion = arm_b.formation_trace.get("lesion_receipt")
        b_source_catalog = arm_b.formation_trace.get("source_evidence_catalog")
        b_forwarded_catalog = arm_b.formation_trace.get(
            "forwarded_evidence_catalog"
        )
        lesioned_grounding_keys = [
            key for key in cited_keys if key.startswith(("state-", "memory-"))
        ]
        retained_anchor_keys = [
            key for key in cited_keys if key not in lesioned_grounding_keys
        ]
        expected_removed_entries = (
            {key: b_source_catalog[key] for key in lesioned_grounding_keys}
            if type(b_source_catalog) is dict
            and all(key in b_source_catalog for key in lesioned_grounding_keys)
            else None
        )
        removed_evidence_refs = (
            []
            if expected_removed_entries is None
            else list(expected_removed_entries.values())
        )
        b_refs = (
            [] if arm_b.formation is None
            else arm_b.formation.get("formation_evidence_refs", [])
        )
        formation_identity_fields = (
            "contract",
            "target_contract_revision",
            "target_model_ref",
        )
        formation_contract_identity_exact = all(
            (
                not any(
                    arm.formation is not None and field in arm.formation
                    for arm in (arm_a, arm_b, arm_c)
                )
            )
            or (
                all(
                    arm.formation is not None and field in arm.formation
                    for arm in (arm_a, arm_b, arm_c)
                )
                and len(
                    {
                        _canonical(arm.formation[field])
                        for arm in (arm_a, arm_b, arm_c)
                        if arm.formation is not None
                    }
                )
                == 1
            )
            for field in formation_identity_fields
        )
        temporal_refs = {
            None if arm.formation is None else arm.formation.get("temporal_sample_ref")
            for arm in (arm_a, arm_b, arm_c)
        }
        initial_states = {arm.before_state for arm in (arm_a, arm_b, arm_c)}
        initial_dbs = {
            _canonical(arm.before_db) for arm in (arm_a, arm_b, arm_c)
        }
        source_unchanged = state_tree_manifest(sealed_source_state) == source_before

        integrity_gates: dict[str, bool] = {
            "fresh_identity_preserves_r1_r2_r3": _preserved_identity_gate(qualification_id),
            "sealed_source_unchanged": source_unchanged,
            "initial_database_bytes_exact": len(set(initial_bytes.values())) == 1,
            "initial_canonical_states_exact": len(initial_states) == 1,
            "initial_episode_projection_histories_exact": len(initial_dbs) == 1,
            "wake_rearm_exact_and_equal": len({_canonical(item) for item in rearm.values()}) == 1
            and all(item.get("noncheckpoint_unchanged") is True for item in rearm.values()),
            "fixed_temporal_ref_exact": len(temporal_refs) == 1 and None not in temporal_refs,
            "formation_contract_identity_exact": formation_contract_identity_exact,
            "a_request_well_formed": a_request is not None,
            "a_empty_target_is_exactly_quiescent": a_request is not None
            and (bool(a_request.strip()) or all(a_quiescent_gates.values())),
            "a_b_full_registration_exact": arm_a.registrations == arm_b.registrations
            and a_map == b_map
            and AUTHORED_ARTIFACT_AFFORDANCE_ID in a_map,
            "c_registration_is_exact_artifact_removal": c_map == expected_c_map
            and arm_c.registrations == expected_c_registrations
            and AUTHORED_ARTIFACT_AFFORDANCE_ID not in c_map,
            "all_registered_operations_effectless": all(
                item["affordance"]["external_effect"] is False
                for arm in (arm_a, arm_b, arm_c)
                for item in arm.registrations
            ),
            "all_restart_exact": all(restart_gates.values()),
            **{f"a_{key}": value for key, value in a_formation_gates.items()},
            **{f"b_{key}": value for key, value in b_formation_gates.items()},
            **{f"c_{key}": value for key, value in c_formation_gates.items()},
            "a_used_real_formation_backend": arm_a.formation_trace.get("original_method_calls") == 1
            and arm_a.formation_trace.get("formation_backend_calls") in (1, 2)
            and arm_a.formation_trace.get("proposal_phase_backend_calls")
            == arm_a.formation_trace.get("formation_backend_calls"),
            "b_used_real_formation_backend": arm_b.formation_trace.get("original_method_calls") == 1
            and arm_b.formation_trace.get("formation_backend_calls") in (1, 2)
            and arm_b.formation_trace.get("proposal_phase_backend_calls")
            == arm_b.formation_trace.get("formation_backend_calls"),
            "b_lesion_record_rebound": arm_b.formation_trace.get("record_rebound") is True,
            "b_lesion_formation_only": arm_b.formation_trace.get(
                "formation_handoff_rebound"
            ) is True
            and arm_b.formation_trace.get("source_input_ref")
            == arm_a.formation_trace.get("source_input_ref")
            and b_source_catalog
            == arm_a.formation_trace.get("source_evidence_catalog"),
            "b_lesion_exact": type(lesion) is dict
            and lesion.get("cited_keys") == cited_keys
            and lesion.get("lesioned_grounding_keys") == lesioned_grounding_keys
            and lesion.get("retained_anchor_keys") == retained_anchor_keys
            and lesion.get("removed_catalog_entries") == expected_removed_entries
            and lesion.get("removed_evidence_refs") == removed_evidence_refs
            and lesion.get("source_formation_input_ref")
            == arm_b.formation_trace.get("source_input_ref")
            and lesion.get("forwarded_formation_input_ref")
            == arm_b.formation_trace.get("forwarded_input_ref")
            and lesion.get("temporal_unchanged") is True
            and lesion.get("anchors_unchanged") is True
            and lesion.get("catalog_exact") is True
            and lesion.get("state_payload_exact") is True
            and lesion.get("memory_payload_exact") is True
            and type(b_forwarded_catalog) is dict
            and lesion.get("forwarded_catalog_ref")
            == _content_ref(b_forwarded_catalog)
            and (
                (
                    bool(lesioned_grounding_keys)
                    and arm_b.formation_trace.get("source_input_ref")
                    != arm_b.formation_trace.get("forwarded_input_ref")
                )
                or (
                    not lesioned_grounding_keys
                    and arm_b.formation_trace.get("source_input_ref")
                    == arm_b.formation_trace.get("forwarded_input_ref")
                )
            ),
            "b_cannot_recite_removed_evidence": type(b_refs) is list
            and not set(removed_evidence_refs).intersection(b_refs),
            "zero_lesion_is_behaviorally_equivalent": (
                _zero_lesion_is_behaviorally_equivalent(
                    lesioned_grounding_keys=lesioned_grounding_keys,
                    a_request=a_request,
                    b_request=b_request,
                    a_quiescent=all(a_quiescent_gates.values()),
                    b_quiescent=all(b_quiescent_gates.values()),
                    a_formation=arm_a.formation,
                    b_formation=arm_b.formation,
                )
            ),
            "c_rejoined_a_intact_formation_input": arm_c.formation_trace.get("source_input_ref")
            == arm_a.formation_trace.get("source_input_ref"),
            "c_replayed_a_exact_formation": arm_c.formation == arm_a.formation
            and arm_c.formation_trace.get("replay_exact") is True,
            "c_zero_formation_backend_call": arm_c.formation_trace.get("original_method_calls") == 0
            and arm_c.formation_trace.get("formation_backend_calls") == 0
            and arm_c.formation_trace.get("proposal_phase_backend_calls") == 0,
        }
        causal_gates: dict[str, bool] = {
            **{f"a_artifact_{key}": value for key, value in a_artifact_gates.items()},
            "a_target_nonempty_and_cites_evidence": bool(
                a_request is not None and a_request.strip() and cited_keys
            ),
            "b_exact_target_changed_or_became_null": a_request is not None
            and b_request is not None
            and b_request.strip() != a_request.strip(),
            "c_did_not_select_removed_artifact": arm_c.result.get("selected_affordance_id")
            != AUTHORED_ARTIFACT_AFFORDANCE_ID,
            "c_created_no_authored_artifact": arm_c.after_artifacts
            == arm_c.before_artifacts,
            "c_credit_unchanged": arm_c.after_credit == arm_c.before_credit,
        }
        disposition, passed = _classify(
            integrity_gates=integrity_gates,
            causal_gates=causal_gates,
            a_request=a_request,
            a_quiescent=all(a_quiescent_gates.values()),
            a_selected_artifact=a_artifact_gates["selected_authored_artifact"],
        )

        def arm_payload(arm: ArmCapture) -> dict[str, object]:
            return {
                "result": arm.result,
                "formation": arm.formation,
                "formation_trace": arm.formation_trace,
                "before_status": arm.before_status,
                "after_status": arm.after_status,
                "registration_manifest": arm.registrations,
                "registration_manifest_ref": _content_ref(arm.registrations),
                "artifact_count_before": len(arm.before_artifacts),
                "artifact_count_after": len(arm.after_artifacts),
            }

        payload: dict[str, object] = {
            "schema": SCHEMA,
            "qualification_id": qualification_id,
            "disposition": disposition,
            "passed": passed,
            "preserved_consumed_predecessors": PRESERVED_CONSUMED_IDENTITIES,
            "inputs": {
                "live_state_root_read_only_guard": str(live_state_root),
                "sealed_source_manifest_ref": _content_ref(source_before),
                "binding_path": str(binding_path),
                "binding_file_sha256": sha256(binding_path.read_bytes()).hexdigest(),
                "binding_ref": binding.binding_ref,
                "library_root": str(library_root),
                "runtime_ref": binding.runtime_ref,
                "selector_qualification_ref": binding.selector_qualification_ref,
                "served_model": served_model,
                "fixed_temporal_input": fixed_temporal.payload(),
                "runtime_sources": _runtime_source_identities(),
            },
            "intervention": {
                "a": "intact formation evidence plus authored-artifact registration",
                "b": (
                    "remove A-cited evidence during target formation only; retain "
                    "artifact registration"
                ),
                "c": (
                    "replay A exact frozen target with zero formation backend call; "
                    "remove artifact registration"
                ),
                "topic_request_or_artifact_supplied_by_harness": False,
                "canonical_state_mutated_before_arm": False,
                "wake_checkpoint_rearmed_in_clones_only": True,
            },
            "arms": {
                "a": arm_payload(arm_a),
                "b": arm_payload(arm_b),
                "c": arm_payload(arm_c),
            },
            "artifact": artifact,
            "wake_rearm": rearm,
            "restart_gates": restart_gates,
            "integrity_gates": integrity_gates,
            "causal_gates": causal_gates,
            "honest_null_gates": a_quiescent_gates,
            "source_state_unchanged": source_unchanged,
            "wall_time_seconds": time.perf_counter() - started,
            "nonclaims": {
                "capability_promotion": False,
                "utility_credit": False,
                "external_effect_authority": False,
                "broad_autonomy_personhood_feeling_or_consciousness": False,
            },
        }
        if not source_unchanged:
            payload["disposition"] = "INVALID"
            payload["passed"] = False
        path, digest = _write_result(result_root, payload)
        return path, digest, str(payload["disposition"])
    finally:
        for runtime in runtimes.values():
            try:
                runtime.close()  # type: ignore[attr-defined]
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-state-root", type=Path, required=True)
    parser.add_argument("--sealed-source-state", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument(
        "--served-model",
        help="optional exact served-model assertion; defaults to the sealed binding",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--qualification-id", required=True)
    args = parser.parse_args()
    result_root_preexisting = args.result_root.exists()
    try:
        path, digest, disposition = run_qualification(
            live_state_root=args.live_state_root,
            sealed_source_state=args.sealed_source_state,
            binding_path=args.binding,
            library_root=args.library_root,
            served_model=args.served_model,
            work_root=args.work_root,
            result_root=args.result_root,
            qualification_id=args.qualification_id,
        )
    except Exception as exc:
        summary: dict[str, object] = {
            "schema": SCHEMA,
            "disposition": "INVALID",
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)[:2_048]},
        }
        if not result_root_preexisting and args.result_root.is_dir():
            invalid = {
                "schema": SCHEMA,
                "qualification_id": args.qualification_id,
                "disposition": "INVALID",
                "passed": False,
                "preserved_consumed_predecessors": PRESERVED_CONSUMED_IDENTITIES,
                "error": summary["error"],
            }
            path, digest = _write_result(args.result_root, invalid)
            summary.update({"path": str(path), "sha256": digest})
        print(json.dumps(summary, sort_keys=True))
        return 2
    passed = disposition == "PASS"
    print(
        json.dumps(
            {
                "disposition": disposition,
                "passed": passed,
                "path": str(path),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 2 if disposition == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
