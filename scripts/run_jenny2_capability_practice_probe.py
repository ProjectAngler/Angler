#!/usr/bin/env python3
"""Probe userless capability-aware internal inquiry on an immutable state clone."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
import subprocess

from angler.memory.cognee_worker_protocol import (
    CogneeWorkerScope,
    SCOPE_MARKER_FILENAME,
    scope_marker_bytes,
    validate_scope_marker,
)
from angler.memory.cognee_state_clone import rebase_cloned_cognee_state_roots
from angler.runtime.jenny2_runtime import assemble_qwen38_autonomous_jenny2_with_cognee


RUNTIME_REF = "sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe"
QUALIFICATION_REF = "sha256:6b06aa8ee734ec405708a24a24b3bd24a762100ebc7fe8761307e24623ea0a73"
SOURCE_IDENTITY = "jenny2-capability-metacognition-v1"


def _require_disjoint_roots(*roots: Path) -> None:
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("source, state, and result roots must be disjoint")


def _require_source_offline(source: Path) -> None:
    service = subprocess.run(
        ("systemctl", "is-active", "--quiet", "jenny2-api.service"),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if service.returncode == 0:
        raise RuntimeError("refusing to clone while Jenny's API service is active")
    if service.returncode != 3:
        raise RuntimeError("cannot establish that Jenny's API service is inactive")
    holders: set[int] = set()
    for process in Path("/proc").glob("[0-9]*"):
        descriptors = process / "fd"
        try:
            entries = tuple(descriptors.iterdir())
        except OSError:
            continue
        for descriptor in entries:
            try:
                target = descriptor.resolve(strict=True)
            except OSError:
                continue
            if target == source or source in target.parents:
                holders.add(int(process.name))
                break
    if holders:
        raise RuntimeError("refusing to clone a state root held by a live process")


def _state_tree_manifest(root: Path) -> tuple[tuple[str, str], ...]:
    try:
        descendants = tuple(root.rglob("*"))
    except OSError as exc:
        raise RuntimeError("source state cannot be inventoried") from exc
    manifest: list[tuple[str, str]] = []
    for candidate in (root, *sorted(descendants, key=lambda item: str(item))):
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise RuntimeError("source state cannot be inventoried") from exc
        relative = "." if candidate == root else candidate.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("source state contains a symbolic link")
        if stat.S_ISDIR(metadata.st_mode):
            manifest.append((relative, "directory"))
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("source state contains a special file")
        digest = sha256()
        try:
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise RuntimeError("source state cannot be inventoried") from exc
        manifest.append((relative, f"file:{digest.hexdigest()}"))
    return tuple(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--identity-suffix", required=True)
    args = parser.parse_args()
    source = Path(args.source_state_root).resolve()
    state_root = Path(args.state_root).resolve()
    result_root = Path(args.result_root).resolve()
    if not (source / "jenny2.sqlite3").is_file() or not (source / "cognee").is_dir():
        raise FileNotFoundError("source learned state is incomplete")
    _require_disjoint_roots(source, state_root, result_root)
    _require_source_offline(source)
    source_scope = CogneeWorkerScope(
        dataset_name=SOURCE_IDENTITY,
        tenant_name=f"{SOURCE_IDENTITY}-tenant",
        node_set_name=f"{SOURCE_IDENTITY}-records",
        state_root=str(source / "cognee"),
    )
    validate_scope_marker(source_scope)
    state_root.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)
    clone = state_root / "clone"
    try:
        source_before = _state_tree_manifest(source)
        shutil.copytree(source, clone, copy_function=shutil.copy2)
        source_after = _state_tree_manifest(source)
        clone_before_rebase = _state_tree_manifest(clone)
        if source_before != source_after or clone_before_rebase != source_after:
            raise RuntimeError("source state changed or cloned inconsistently")
        _require_source_offline(source)
        rebase_cloned_cognee_state_roots(source, clone)
    except BaseException:
        if clone.parent == state_root and clone.name == "clone":
            try:
                clone_metadata = clone.lstat()
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISDIR(clone_metadata.st_mode) and not stat.S_ISLNK(
                    clone_metadata.st_mode
                ):
                    shutil.rmtree(clone)
                else:
                    clone.unlink()
        raise
    scope = CogneeWorkerScope(
        dataset_name=SOURCE_IDENTITY,
        tenant_name=f"{SOURCE_IDENTITY}-tenant",
        node_set_name=f"{SOURCE_IDENTITY}-records",
        state_root=str(clone / "cognee"),
    )
    marker = clone / "cognee" / SCOPE_MARKER_FILENAME
    marker.write_bytes(scope_marker_bytes(scope))
    marker.chmod(0o600)
    runtime = assemble_qwen38_autonomous_jenny2_with_cognee(
        clone / "jenny2.sqlite3",
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=scope,
        runtime_ref=RUNTIME_REF,
        selector_qualification_ref=QUALIFICATION_REF,
    )
    try:
        before = asdict(runtime.status())
        state_before = json.loads(runtime.supervisor.state_bytes())
        trigger = f"jenny2-capability-practice-{args.identity_suffix}:scheduler:0"
        result = runtime.supervisor.scheduler_tick(trigger)
        if result.status == "COMMITTED":
            runtime.supervisor.retry_pending_projections(runtime.projector)
        after = asdict(runtime.status())
        episode_payload = None
        output = None
        if result.episode_ref is not None:
            episode = next(
                item
                for item in runtime.supervisor.episode_items(limit=256)
                if item.episode_ref == result.episode_ref
            )
            episode_payload = json.loads(episode.payload_json)
            output = episode_payload.get("receipt", {}).get("output")
        state_after = json.loads(runtime.supervisor.state_bytes())
        pending_payload = None
        pending_bytes = runtime.supervisor.pending_bytes()
        if pending_bytes is not None:
            pending_payload = json.loads(pending_bytes)
        choice = (
            episode_payload.get("choice")
            if episode_payload is not None
            else None if pending_payload is None else pending_payload.get("choice")
        )
        context = None
        if isinstance(choice, dict):
            context = json.loads(choice["context_json"])
        capability_count = len(state_before.get("capability_evidence", []))
        context_capability_count = (
            0
            if context is None
            else len(
                context.get("cognitive_state", {}).get("capability_evidence", [])
            )
        )
        route_contract_ok = False
        if result.selected_affordance_id == "cortex.respond":
            route_contract_ok = (
                result.status == "COMMITTED"
                and isinstance(choice, dict)
                and bool(choice.get("action_payload"))
                and episode_payload is not None
                and episode_payload.get("observation", {}).get("source")
                == "SCHEDULER"
                and episode_payload.get("receipt", {}).get("status")
                == "COMPLETED_UNEVALUATED"
                and state_before == state_after
            )
        elif result.selected_affordance_id == "control.wait":
            route_contract_ok = (
                result.status == "WAITING"
                and isinstance(choice, dict)
                and choice.get("action_payload") == ""
                and pending_payload is not None
                and state_before == state_after
            )
        elif result.selected_affordance_id == "control.ask":
            route_contract_ok = (
                result.status == "ASKING"
                and isinstance(choice, dict)
                and bool(choice.get("action_payload"))
                and pending_payload is not None
                and state_before == state_after
            )
        elif result.selected_affordance_id == "control.stop":
            route_contract_ok = (
                result.status == "STOPPED" and not after["scheduler_enabled"]
            )
        evidence = {
            "schema": "jenny2.capability-practice-probe.v1",
            "identity": f"jenny2-capability-practice-{args.identity_suffix}",
            "source_identity": SOURCE_IDENTITY,
            "before": before,
            "result": asdict(result),
            "after": after,
            "capability_evidence_count": capability_count,
            "choice": choice,
            "pending": pending_payload,
            "output": output,
            "context_capability_evidence_count": context_capability_count,
            "learned_state_changed": state_before != state_after,
            "behavioral_choice": result.selected_affordance_id,
            "route_contract_ok": route_contract_ok,
            "passed": (
                capability_count >= 2
                and context is not None
                and context_capability_count == capability_count
                and isinstance(choice, dict)
                and route_contract_ok
                and not after["external_effects_enabled"]
            ),
            "nonclaims": [
                "one model-authored internal inquiry is not broad life autonomy",
                "an unevaluated action cannot increase competence evidence",
                "the probe does not establish subjective self-awareness",
            ],
        }
    finally:
        runtime.close()
    result_path = result_root / "result.json"
    result_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("JENNY2_CAPABILITY_PRACTICE=" + json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
