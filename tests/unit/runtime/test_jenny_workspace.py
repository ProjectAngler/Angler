from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import angler.runtime.jenny_workspace as workspace_module
from angler.runtime.jenny_workspace import (
    JennyProjectWorkspaceExecutor,
    JennyWorkspaceExecutor,
    PROJECT_WORKSPACE_ACTION_CONTRACT,
    PROJECT_WORKSPACE_AFFORDANCE,
    PROJECT_WORKSPACE_AFFORDANCE_ID,
    PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
    PROJECT_WORKSPACE_PERMISSION_SCOPE,
    WorkspaceBoundsError,
    WorkspaceConflictError,
    WorkspaceIntegrityError,
    WorkspaceSecurityError,
    workspace_content_ref,
)
from angler.runtime.persistent_autonomy import (
    AffordanceReceipt,
    AffordanceRequest,
    ObservableConsequence,
)


def _ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request(
    payload: dict[str, object], *, key: str = "request-1"
) -> AffordanceRequest:
    return AffordanceRequest(
        idempotency_key=_ref(key),
        trigger_ref="test.workspace",
        affordance_id=PROJECT_WORKSPACE_AFFORDANCE_ID,
        observation_ref=_ref("observation"),
        state_head_ref=_ref("state"),
        action_payload=json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _list_action(
    path: str = ".", *, after: str | None = None, maximum: int = 64
) -> dict[str, object]:
    return {
        "after": after,
        "contract": PROJECT_WORKSPACE_ACTION_CONTRACT,
        "max_entries": maximum,
        "operation": "list",
        "path": path,
    }


def _read_action(
    path: str,
    *,
    cursor: int = 0,
    maximum: int = 8_192,
    expected: str | None = None,
) -> dict[str, object]:
    return {
        "contract": PROJECT_WORKSPACE_ACTION_CONTRACT,
        "cursor": cursor,
        "expected_content_ref": expected,
        "max_chars": maximum,
        "operation": "read",
        "path": path,
    }


def _write_action(
    path: str,
    content: str,
    *,
    mode: str,
    expected: str | None,
    declared_ref: str | None = None,
) -> dict[str, object]:
    return {
        "content": content,
        "content_ref": declared_ref or workspace_content_ref(content),
        "contract": PROJECT_WORKSPACE_ACTION_CONTRACT,
        "expected_content_ref": expected,
        "mode": mode,
        "operation": "write",
        "path": path,
    }


def _mkdir_action(path: str) -> dict[str, object]:
    return {
        "contract": PROJECT_WORKSPACE_ACTION_CONTRACT,
        "mode": "create",
        "operation": "mkdir",
        "path": path,
    }


def _observation(receipt: AffordanceReceipt) -> dict[str, object]:
    assert isinstance(receipt.observable_consequence, ObservableConsequence)
    return json.loads(receipt.observable_consequence.observation_json)


def test_exact_actions_create_read_list_and_cas_replace(tmp_path: Path) -> None:
    with JennyWorkspaceExecutor(tmp_path) as executor:
        assert JennyProjectWorkspaceExecutor is JennyWorkspaceExecutor
        assert PROJECT_WORKSPACE_AFFORDANCE.affordance_id == executor.AFFORDANCE_ID
        assert PROJECT_WORKSPACE_AFFORDANCE.permission_scope == (
            PROJECT_WORKSPACE_PERMISSION_SCOPE
        )
        assert PROJECT_WORKSPACE_AFFORDANCE.external_effect is False

        mkdir_receipt = executor(_request(_mkdir_action("src"), key="mkdir"))
        assert mkdir_receipt.status == "COMPLETED"
        assert (tmp_path / "src").is_dir()

        original = "hello, Jenny\n"
        original_ref = workspace_content_ref(original)
        create_receipt = executor(
            _request(
                _write_action(
                    "src/main.txt", original, mode="create", expected=None
                ),
                key="create",
            )
        )
        assert (tmp_path / "src/main.txt").read_text(encoding="utf-8") == original
        created = _observation(create_receipt)
        assert created == {
            "content_ref": original_ref,
            "contract": PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
            "mode": "create",
            "operation": "write",
            "path": "src/main.txt",
            "previous_content_ref": None,
            "size_bytes": len(original.encode("utf-8")),
            "workspace_ref": executor.workspace_ref,
        }
        assert create_receipt.consequence == ()
        assert create_receipt.observable_consequence is not None
        assert create_receipt.observable_consequence.source_kind == "TOOL"
        assert create_receipt.observable_consequence.source_ref == executor.workspace_ref
        assert original_ref in create_receipt.observable_consequence.artifact_refs
        assert executor.workspace_ref in (
            create_receipt.observable_consequence.evidence_refs
        )

        read_receipt = executor(
            _request(
                _read_action("src/main.txt", maximum=5, expected=original_ref),
                key="read",
            )
        )
        read = _observation(read_receipt)
        assert read["content"] == "hello"
        assert read["content_ref"] == original_ref
        assert read["span"] == {"end": 5, "start": 0}
        assert read["span_ref"] == workspace_content_ref("hello")
        assert read["eof"] is False

        listing_receipt = executor(
            _request(_list_action("src", maximum=8), key="list")
        )
        listing = _observation(listing_receipt)
        assert listing["entries"] == [
            {
                "kind": "file",
                "name": "main.txt",
                "path": "src/main.txt",
                "size_bytes": len(original.encode("utf-8")),
            }
        ]
        assert listing["eof"] is True
        assert listing["listing_ref"] in (
            listing_receipt.observable_consequence.artifact_refs  # type: ignore[union-attr]
        )

        replacement = "revised text\n"
        replace_receipt = executor(
            _request(
                _write_action(
                    "src/main.txt",
                    replacement,
                    mode="replace",
                    expected=original_ref,
                ),
                key="replace",
            )
        )
        replaced = _observation(replace_receipt)
        assert replaced["previous_content_ref"] == original_ref
        assert replaced["content_ref"] == workspace_content_ref(replacement)
        assert (tmp_path / "src/main.txt").read_text(encoding="utf-8") == replacement
        assert not os.access(tmp_path / "src/main.txt", os.X_OK)


def test_create_and_replace_modes_fail_closed_on_conflicts(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("one", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        with pytest.raises(WorkspaceConflictError, match="already exists"):
            executor(
                _request(
                    _write_action("note.txt", "two", mode="create", expected=None),
                    key="create-conflict",
                )
            )
        assert target.read_text(encoding="utf-8") == "one"

        stale = workspace_content_ref("stale")
        with pytest.raises(WorkspaceConflictError, match="stale"):
            executor(
                _request(
                    _write_action(
                        "note.txt", "two", mode="replace", expected=stale
                    ),
                    key="replace-conflict",
                )
            )
        assert target.read_text(encoding="utf-8") == "one"

        malformed_create = _write_action(
            "new.txt", "new", mode="create", expected=workspace_content_ref("")
        )
        with pytest.raises(ValueError, match="null expected_content_ref"):
            executor(_request(malformed_create, key="create-has-cas"))
        assert not (tmp_path / "new.txt").exists()

        malformed_replace = _write_action(
            "note.txt", "two", mode="replace", expected=None
        )
        with pytest.raises(ValueError, match="expected_content_ref"):
            executor(_request(malformed_replace, key="replace-no-cas"))
        assert target.read_text(encoding="utf-8") == "one"


@pytest.mark.parametrize(
    "path",
    (
        "../outside.txt",
        "/tmp/outside.txt",
        "sub/../../outside.txt",
        "sub//file.txt",
        "./file.txt",
        "sub\\file.txt",
        "bad\x00name",
    ),
)
def test_lexical_escape_is_rejected_without_outside_access(
    tmp_path: Path, path: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with JennyWorkspaceExecutor(workspace) as executor:
        with pytest.raises(WorkspaceSecurityError):
            executor(
                _request(
                    _write_action(path, "changed", mode="create", expected=None),
                    key=f"escape-{path}",
                )
            )
    assert outside.read_text(encoding="utf-8") == "outside"


def test_symlinks_and_hardlinks_never_reach_outside_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (workspace / "linked-dir").symlink_to(outside, target_is_directory=True)
    (workspace / "linked-file").symlink_to(secret)
    os.link(secret, workspace / "hard-file")

    with JennyWorkspaceExecutor(workspace) as executor:
        for index, path in enumerate(
            ("linked-dir/secret.txt", "linked-file", "hard-file")
        ):
            with pytest.raises(WorkspaceSecurityError):
                executor(_request(_read_action(path), key=f"linked-read-{index}"))
        with pytest.raises(WorkspaceSecurityError):
            executor(
                _request(
                    _write_action(
                        "linked-file",
                        "changed",
                        mode="replace",
                        expected=workspace_content_ref("secret"),
                    ),
                    key="linked-write",
                )
            )

        listing = _observation(executor(_request(_list_action(), key="links-list")))
        listed = {item["name"]: item["kind"] for item in listing["entries"]}
        assert listed["linked-dir"] == "symlink-unavailable"
        assert listed["linked-file"] == "symlink-unavailable"
        assert listed["hard-file"] == "unsupported"
    assert secret.read_text(encoding="utf-8") == "secret"


def test_root_must_be_canonical_real_and_same_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(WorkspaceSecurityError, match="symlink"):
        JennyWorkspaceExecutor(alias)
    with pytest.raises(ValueError, match="canonical"):
        JennyWorkspaceExecutor(str(real / ".." / "real"))
    monkeypatch.setattr(workspace_module.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(WorkspaceSecurityError, match="owned"):
        JennyWorkspaceExecutor(real)


def test_root_inode_replacement_invalidates_the_executor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    moved = tmp_path / "moved"
    workspace.mkdir()
    executor = JennyWorkspaceExecutor(workspace)
    try:
        workspace.rename(moved)
        workspace.mkdir()
        with pytest.raises(WorkspaceIntegrityError, match="root binding changed"):
            executor(_request(_list_action(), key="rebound-root"))
    finally:
        executor.close()


def test_replace_detects_a_race_after_staging_and_does_not_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "race.txt"
    target.write_text("initial", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        original_stage = executor._stage_content

        def race_after_stage(parent_descriptor: int, content: bytes) -> str:
            staged_name = original_stage(parent_descriptor, content)
            target.write_text("racer", encoding="utf-8")
            return staged_name

        monkeypatch.setattr(executor, "_stage_content", race_after_stage)
        with pytest.raises(WorkspaceConflictError, match="changed before commit"):
            executor(
                _request(
                    _write_action(
                        "race.txt",
                        "ours",
                        mode="replace",
                        expected=workspace_content_ref("initial"),
                    ),
                    key="race",
                )
            )
    assert target.read_text(encoding="utf-8") == "racer"
    assert not any(
        item.name.startswith(".__jenny_workspace_staged__")
        for item in tmp_path.iterdir()
    )


def test_declared_content_ref_detects_tampering_before_any_write(
    tmp_path: Path,
) -> None:
    with JennyWorkspaceExecutor(tmp_path) as executor:
        with pytest.raises(WorkspaceIntegrityError, match="declared content_ref"):
            executor(
                _request(
                    _write_action(
                        "tampered.txt",
                        "actual",
                        mode="create",
                        expected=None,
                        declared_ref=workspace_content_ref("different"),
                    ),
                    key="tampered",
                )
            )
    assert not (tmp_path / "tampered.txt").exists()


def test_replace_is_fsync_then_atomic_replace_and_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "atomic.txt"
    target.write_text("old", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        real_fsync = workspace_module.os.fsync
        real_replace = workspace_module.os.replace
        events: list[str] = []

        def recording_fsync(descriptor: int) -> None:
            events.append("fsync")
            real_fsync(descriptor)

        def recording_replace(*args: object, **kwargs: object) -> None:
            events.append("replace")
            real_replace(*args, **kwargs)

        monkeypatch.setattr(workspace_module.os, "fsync", recording_fsync)
        monkeypatch.setattr(workspace_module.os, "replace", recording_replace)
        executor(
            _request(
                _write_action(
                    "atomic.txt",
                    "new complete content",
                    mode="replace",
                    expected=workspace_content_ref("old"),
                ),
                key="atomic-success",
            )
        )
        replace_index = events.index("replace")
        assert "fsync" in events[:replace_index]
        assert "fsync" in events[replace_index + 1 :]
        assert target.read_text(encoding="utf-8") == "new complete content"

    target.write_text("stable", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        def failed_replace(*_: object, **__: object) -> None:
            raise OSError("injected replace failure")

        monkeypatch.setattr(workspace_module.os, "replace", failed_replace)
        with pytest.raises(WorkspaceIntegrityError, match="before installation"):
            executor(
                _request(
                    _write_action(
                        "atomic.txt",
                        "must not appear",
                        mode="replace",
                        expected=workspace_content_ref("stable"),
                    ),
                    key="atomic-failure",
                )
            )
    assert target.read_text(encoding="utf-8") == "stable"
    assert not any(
        item.name.startswith(".__jenny_workspace_staged__")
        for item in tmp_path.iterdir()
    )


def test_requests_files_reads_responses_and_directory_scans_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with JennyWorkspaceExecutor(
        tmp_path,
        max_file_bytes=8,
        max_read_chars=3,
        max_list_entries=2,
    ) as executor:
        with pytest.raises(WorkspaceBoundsError, match="file byte ceiling"):
            executor(
                _request(
                    _write_action(
                        "large.txt", "123456789", mode="create", expected=None
                    ),
                    key="large-write",
                )
            )
        (tmp_path / "large.txt").write_text("123456789", encoding="utf-8")
        with pytest.raises(WorkspaceBoundsError, match="file exceeds"):
            executor(
                _request(
                    _read_action("large.txt", maximum=3), key="large-read"
                )
            )
        with pytest.raises(WorkspaceBoundsError, match="max_chars"):
            executor(
                _request(_read_action("large.txt", maximum=4), key="read-window")
            )
        with pytest.raises(WorkspaceBoundsError, match="max_entries"):
            executor(_request(_list_action(maximum=3), key="list-page"))

    with JennyWorkspaceExecutor(tmp_path, max_file_bytes=1024 * 1024) as executor:
        oversized_bytes = "🙂" * 4_100
        request = _request(
            _write_action(
                "request.txt", oversized_bytes, mode="create", expected=None
            ),
            key="request-bytes",
        )
        assert len(request.action_payload) < 16_384
        assert len(request.action_payload.encode("utf-8")) > 16_384
        with pytest.raises(WorkspaceBoundsError, match="action exceeds"):
            executor(request)

    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    for name in ("a", "b", "c"):
        (scan_root / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(workspace_module, "_MAX_DIRECTORY_SCAN_ENTRIES", 2)
    with JennyWorkspaceExecutor(scan_root) as executor:
        with pytest.raises(WorkspaceBoundsError, match="scan ceiling"):
            executor(_request(_list_action(), key="scan-bound"))


def test_large_multibyte_read_is_shrunk_to_bounded_utf8_response(
    tmp_path: Path,
) -> None:
    content = "🙂" * 8_192
    (tmp_path / "unicode.txt").write_text(content, encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        receipt = executor(
            _request(
                _read_action("unicode.txt", maximum=8_192),
                key="unicode-read",
            )
        )
        assert receipt.observable_consequence is not None
        encoded = receipt.observable_consequence.observation_json.encode("utf-8")
        assert len(encoded) <= workspace_module._MAX_OBSERVATION_BYTES
        observed = _observation(receipt)
        assert 0 < len(observed["content"]) < len(content)
        assert observed["eof"] is False
        assert observed["span_ref"] == workspace_content_ref(observed["content"])
        assert observed["content_ref"] == workspace_content_ref(content)


def test_invalid_utf8_and_read_content_conflict_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")
    (tmp_path / "text.txt").write_text("current", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        with pytest.raises(WorkspaceIntegrityError, match="not valid UTF-8"):
            executor(_request(_read_action("binary.dat"), key="binary"))
        with pytest.raises(WorkspaceConflictError, match="differs"):
            executor(
                _request(
                    _read_action(
                        "text.txt", expected=workspace_content_ref("previous")
                    ),
                    key="read-cas",
                )
            )


def test_identical_reservation_replays_but_detects_reuse_and_later_tamper(
    tmp_path: Path,
) -> None:
    action = _write_action("replay.txt", "once", mode="create", expected=None)
    request = _request(action, key="same-reservation")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        first = executor(request)
        second = executor(request)
        assert second is first
        assert first.receipt_ref == second.receipt_ref
        assert (tmp_path / "replay.txt").read_text(encoding="utf-8") == "once"

        reused = _request(
            _write_action("other.txt", "other", mode="create", expected=None),
            key="same-reservation",
        )
        with pytest.raises(WorkspaceConflictError, match="reused"):
            executor(reused)
        assert not (tmp_path / "other.txt").exists()

        (tmp_path / "replay.txt").write_text("tampered", encoding="utf-8")
        with pytest.raises(WorkspaceIntegrityError, match="result changed"):
            executor(request)


def test_no_shell_execution_delete_or_rename_operations_exist(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with JennyWorkspaceExecutor(tmp_path) as executor:
        for index, operation in enumerate(("shell", "execute", "delete", "rename")):
            payload = {
                "contract": PROJECT_WORKSPACE_ACTION_CONTRACT,
                "operation": operation,
                "path": "sentinel.txt",
            }
            with pytest.raises(ValueError, match="unavailable"):
                executor(_request(payload, key=f"forbidden-{index}"))
    assert sentinel.read_text(encoding="utf-8") == "preserve"
