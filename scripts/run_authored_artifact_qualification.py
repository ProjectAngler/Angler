#!/usr/bin/env python3
"""Effectless two-arm witness for Jenny's model-authored artifact capability.

The input is an already sealed snapshot of canonical Jenny state, never the live
state directory.  Both arms start from byte-identical canonical state.  The
control differs only by removal of ``internal.authored-artifact`` from the
dynamic affordance catalog.  No request, topic, or artifact is supplied by this
harness; an empty model initiative is an honest, non-mutating outcome.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Mapping

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
import angler.runtime.jenny2_runtime as jenny2_runtime_module


SCHEMA = "jenny2.authored-artifact-cloned-state-qualification.v1"
_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}")
_CREDIT_FIELDS = (
    "affordance_utility",
    "memory_utility",
    "affordance_outcome_profiles",
    "capability_evidence",
    "capability_use_evidence",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_ref(manifest: object) -> str:
    return "sha256:" + sha256(_canonical(manifest)).hexdigest()


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


def _rearm_autonomy_wake(database: Path) -> None:
    """Remove only the clone's mechanical wake-deduplication checkpoint."""

    with closing(sqlite3.connect(database)) as connection:
        before = connection.execute(
            "SELECT state_ref,state_blob,scheduler_enabled,moving_origin_ordinal,"
            "last_event_ref,pending_json,shadow_json,revision "
            "FROM supervisor_state WHERE singleton=1"
        ).fetchone()
        if before is None:
            raise RuntimeError("cloned supervisor state is absent")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM autonomy_wake_checkpoint WHERE singleton=1")
        connection.commit()
        after = connection.execute(
            "SELECT state_ref,state_blob,scheduler_enabled,moving_origin_ordinal,"
            "last_event_ref,pending_json,shadow_json,revision "
            "FROM supervisor_state WHERE singleton=1"
        ).fetchone()
    if after != before:
        raise RuntimeError("re-arming the clone changed canonical state")


def _assemble_arm(
    prepared: PreparedQualificationClone,
    binding: SGLangLoRABinding,
    binding_path: Path,
    served_model: str,
    library_root: Path,
    *,
    include_authored_artifact: bool,
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


def _catalog(runtime: object) -> list[dict[str, object]]:
    return [
        asdict(item)
        for item in runtime.supervisor.affordances.definitions()  # type: ignore[attr-defined]
    ]


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
    catalog: tuple[dict[str, object], ...]


def _capture_arm(runtime: object, *, name: str, trigger_ref: str) -> ArmCapture:
    database = Path(runtime.supervisor.path)  # type: ignore[attr-defined]
    before_status = _status(runtime)
    before_state = runtime.supervisor.state_bytes()  # type: ignore[attr-defined]
    before_payload = _state_payload(runtime)
    before_db = _db_snapshot(database)
    catalog = tuple(_catalog(runtime))
    if any(item["external_effect"] is not False for item in catalog):
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
        catalog=catalog,
    )


def _artifact_gates(arm: ArmCapture) -> tuple[dict[str, bool], dict[str, object] | None]:
    result = arm.result
    selected = (
        result.get("selected_affordance_id") == AUTHORED_ARTIFACT_AFFORDANCE_ID
    )
    artifact: dict[str, object] | None = None
    identity_valid = False
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
        "one_artifact_appended": len(arm.after_artifacts) == len(arm.before_artifacts) + 1,
        "content_identity_valid": identity_valid,
        "one_episode_committed": len(arm.after_db["episodes"]) == len(arm.before_db["episodes"]) + 1,
        "moving_origin_advanced_once": arm.after_status["moving_origin_ordinal"] == arm.before_status["moving_origin_ordinal"] + 1,
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
        "settled": arm.after_status["pending_operation"] is False and arm.after_status["pending_projections"] == 0,
        "external_effects_disabled": arm.after_status["external_effects_enabled"] is False,
    }, artifact


def _quiescent_gates(arm: ArmCapture) -> dict[str, bool]:
    return {
        "honest_quiescent": arm.result.get("status") == "QUIESCENT",
        "canonical_state_unchanged": arm.after_state == arm.before_state,
        "moving_origin_unchanged": arm.after_status["moving_origin_ordinal"] == arm.before_status["moving_origin_ordinal"],
        "episode_history_unchanged": arm.after_db["episodes"] == arm.before_db["episodes"],
        "projection_history_unchanged": arm.after_db["projections"] == arm.before_db["projections"],
        "artifact_history_unchanged": arm.after_artifacts == arm.before_artifacts,
        "credit_unchanged": arm.after_credit == arm.before_credit,
    }


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
) -> tuple[Path, str, str]:
    if _ID.fullmatch(qualification_id) is None:
        raise ValueError("qualification_id must be a bounded lowercase identity")
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
    if not sealed_source_state.is_absolute() or sealed_source_state.resolve(strict=True) != sealed_source_state:
        raise ValueError("sealed source state must be an exact canonical directory")
    if (
        sealed_source_state == live_state_root
        or sealed_source_state in live_state_root.parents
        or live_state_root in sealed_source_state.parents
    ):
        raise ValueError("sealed source must be disjoint from the live state root")
    for output_root in (work_root, result_root):
        if (
            output_root == sealed_source_state
            or output_root in sealed_source_state.parents
            or sealed_source_state in output_root.parents
            or output_root == live_state_root
            or output_root in live_state_root.parents
            or live_state_root in output_root.parents
        ):
            raise ValueError("live, sealed-source, work, and result roots must be disjoint")
    work_root.mkdir(mode=0o700)
    result_root.mkdir(mode=0o700)
    started = time.perf_counter()
    source_before = state_tree_manifest(sealed_source_state)

    def sealed_checker(source: Path) -> None:
        if source != sealed_source_state or state_tree_manifest(source) != source_before:
            raise RuntimeError("sealed source state changed during qualification")

    binding = SGLangLoRABinding.from_file(binding_path)
    if served_model is None:
        served_model = binding.served_model
    if binding.served_model != served_model or binding.selector_qualification_ref is None:
        raise ValueError("served model/binding qualification identity differs")
    if (
        not library_root.is_absolute()
        or library_root.resolve(strict=True) != library_root
        or not library_root.is_dir()
    ):
        raise ValueError("library root must be an exact canonical directory")
    prepared = {
        name: prepare_qualification_clone(
            sealed_source_state,
            work_root / name,
            offline_checker=sealed_checker,
        )
        for name in ("capability_present", "registration_removed")
    }
    databases = {name: item.root / "jenny2.sqlite3" for name, item in prepared.items()}
    initial_bytes = {name: database.read_bytes() for name, database in databases.items()}
    if initial_bytes["capability_present"] != initial_bytes["registration_removed"]:
        raise RuntimeError("qualification clone databases differ before intervention")
    for database in databases.values():
        _rearm_autonomy_wake(database)

    runtimes: dict[str, object] = {}
    captures: dict[str, ArmCapture] = {}
    trigger_ref = qualification_id + ":autonomous-wake"
    try:
        for name, include in (("capability_present", True), ("registration_removed", False)):
            runtime = runtime_factory(
                prepared[name], binding, binding_path, served_model, library_root,
                include_authored_artifact=include,
            )
            runtimes[name] = runtime
            captures[name] = _capture_arm(runtime, name=name, trigger_ref=trigger_ref)
            runtime.close()  # type: ignore[attr-defined]
            del runtimes[name]

        treatment = captures["capability_present"]
        control = captures["registration_removed"]
        treatment_catalog = {item["affordance_id"]: item for item in treatment.catalog}
        control_catalog = {item["affordance_id"]: item for item in control.catalog}
        catalog_fair = bool(
            AUTHORED_ARTIFACT_AFFORDANCE_ID in treatment_catalog
            and AUTHORED_ARTIFACT_AFFORDANCE_ID not in control_catalog
            and {
                key: value for key, value in treatment_catalog.items()
                if key != AUTHORED_ARTIFACT_AFFORDANCE_ID
            } == control_catalog
        )
        artifact_gates, artifact = _artifact_gates(treatment)
        quiescent_gates = _quiescent_gates(treatment)

        restart_gates: dict[str, bool] = {}
        for name, include in (("capability_present", True), ("registration_removed", False)):
            restarted = runtime_factory(
                prepared[name], binding, binding_path, served_model, library_root,
                include_authored_artifact=include,
            )
            runtimes[name] = restarted
            capture = captures[name]
            restart_status = _status(restarted)
            restart_gates[name] = bool(
                restarted.supervisor.state_bytes() == capture.after_state  # type: ignore[attr-defined]
                and all(
                    restart_status[field] == capture.after_status[field]
                    for field in ("state_ref", "last_event_ref", "moving_origin_ordinal")
                )
            )
            if name == "capability_present" and artifact_gates["committed"]:
                replay_before = restarted.supervisor.state_bytes()  # type: ignore[attr-defined]
                replay = restarted.supervisor.scheduler_tick(  # type: ignore[attr-defined]
                    trigger_ref
                )
                restart_gates["treatment_replay_exact"] = bool(
                    replay.episode_ref == treatment.result.get("episode_ref")
                    and restarted.supervisor.state_bytes() == replay_before  # type: ignore[attr-defined]
                    and _status(restarted) == restart_status
                )
            restarted.close()  # type: ignore[attr-defined]
            del runtimes[name]

        control_gates = {
            "catalog_is_exact_removal": catalog_fair,
            "initial_canonical_state_exact": control.before_state == treatment.before_state,
            "no_authored_artifact_created": control.after_artifacts == control.before_artifacts,
            "credit_unchanged": control.after_credit == control.before_credit,
            "external_effects_disabled": control.after_status["external_effects_enabled"] is False,
        }
        if artifact_gates["selected_authored_artifact"]:
            witness_gates = {
                **artifact_gates,
                **{f"restart_{key}": value for key, value in restart_gates.items()},
                **{f"control_{key}": value for key, value in control_gates.items()},
            }
            disposition = "PASS" if all(witness_gates.values()) else "INVALID"
        elif all(quiescent_gates.values()):
            witness_gates = {
                **quiescent_gates,
                "restart_exact": restart_gates.get("capability_present", False),
                **{f"control_{key}": value for key, value in control_gates.items()},
            }
            disposition = "HONEST_QUIESCENT" if all(witness_gates.values()) else "INVALID"
        else:
            witness_gates = {
                "real_model_selection_completed": treatment.result.get("status") in {"COMMITTED", "SHADOW", "WAITING", "ASKING", "STOPPED"},
                "credit_unchanged": treatment.after_credit == treatment.before_credit,
                "restart_exact": restart_gates.get("capability_present", False),
                **{f"control_{key}": value for key, value in control_gates.items()},
            }
            disposition = "NO_ARTIFACT_SELECTED" if all(witness_gates.values()) else "INVALID"

        payload: dict[str, object] = {
            "schema": SCHEMA,
            "qualification_id": qualification_id,
            "disposition": disposition,
            "passed": disposition == "PASS",
            "inputs": {
                "live_state_root_read_only_guard": str(live_state_root),
                "sealed_source_manifest_ref": _manifest_ref(source_before),
                "binding_path": str(binding_path),
                "binding_file_sha256": sha256(binding_path.read_bytes()).hexdigest(),
                "binding_ref": binding.binding_ref,
                "library_root": str(library_root),
                "runtime_ref": binding.runtime_ref,
                "selector_qualification_ref": binding.selector_qualification_ref,
                "served_model": served_model,
            },
            "intervention": {
                "treatment": "normal dynamic catalog",
                "control": f"exact removal of {AUTHORED_ARTIFACT_AFFORDANCE_ID}",
                "topic_or_artifact_supplied_by_harness": False,
                "wake_checkpoint_rearmed_in_clones_only": True,
            },
            "treatment": {
                "result": treatment.result,
                "artifact": artifact,
                "before_status": treatment.before_status,
                "after_status": treatment.after_status,
            },
            "control": {
                "result": control.result,
                "before_status": control.before_status,
                "after_status": control.after_status,
            },
            "gates": witness_gates,
            "source_state_unchanged": state_tree_manifest(sealed_source_state) == source_before,
            "wall_time_seconds": time.perf_counter() - started,
            "nonclaims": {
                "capability_promotion": False,
                "utility_credit": False,
                "external_effect_authority": False,
                "broad_autonomy_or_personhood": False,
            },
        }
        if payload["source_state_unchanged"] is not True:
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
            "disposition": "INVALID",
            "error": {"type": type(exc).__name__, "message": str(exc)[:2_048]},
        }
        if not result_root_preexisting and args.result_root.is_dir():
            invalid = {
                "schema": SCHEMA,
                "qualification_id": args.qualification_id,
                "disposition": "INVALID",
                "passed": False,
                "error": summary["error"],
            }
            path, digest = _write_result(args.result_root, invalid)
            summary.update({"path": str(path), "sha256": digest})
        print(json.dumps(summary, sort_keys=True))
        return 2
    print(
        json.dumps(
            {"disposition": disposition, "path": str(path), "sha256": digest},
            sort_keys=True,
        )
    )
    return (
        0
        if disposition in {"PASS", "HONEST_QUIESCENT", "NO_ARTIFACT_SELECTED"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
