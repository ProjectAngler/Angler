"""Immutable Jenny 2.0 genesis lineage.

The record binds owner-verified Jenny 1.x provenance without reading, loading,
or importing any legacy source or personal state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import re


GENESIS_CONTRACT = "ANG-CTR-JENNY-GENESIS-001@0.1.0"
OWNER_AGENT_REF = "jenny.agent.v2"
OWNER_PRESERVATION_ROOT = (
    r"C:\JennyRecovery\recovered-wsl-jenny-core-2026-09-01\jenny-core"
)
OWNER_ARCHIVE_PATH = (
    "C:\\JennyRecovery\\recovered-wsl-jenny-core-2026-09-01\\"
    "jenny-core-recovered-without-model.tar.zst"
)
OWNER_ARCHIVE_SHA256 = (
    "ac50b66c8701d8ec6b090b1c2b0e9bbd9265b96a83b6cab16c9159eb3077c5db"
)
OWNER_JENNY1_FREEZE = "c87860b22b72c4466c41a53a3c2712c2548aa213"
OWNER_JENNY_NEXT_HEAD = "10d7c14c3d9e4b0c0cc104b52b110864a07b2a49"
OWNER_COMPATIBILITY_SHELL = r"C:\JennyRuntime\source\jenny_next"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_AGENT = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _utc(value: str, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must be UTC")
    return value


@dataclass(frozen=True, slots=True)
class JennyGenesis:
    contract: str
    agent_ref: str
    created_at_utc: str
    preservation_root: str
    archive_path: str
    archive_sha256: str
    jenny1_freeze_commit: str
    jenny_next_head_commit: str
    compatibility_shell_path: str
    runtime_import_status: str = "NOT_IMPORTED"
    personal_state_status: str = "NOT_IMPORTED"
    missions_status: str = "OMITTED"
    consultants_status: str = "OMITTED"

    def __post_init__(self) -> None:
        if self.contract != GENESIS_CONTRACT:
            raise ValueError("unsupported Jenny genesis contract")
        if type(self.agent_ref) is not str or not _AGENT.fullmatch(self.agent_ref):
            raise ValueError("agent_ref must be a stable lowercase operational reference")
        _utc(self.created_at_utc, "created_at_utc")
        for label, value in (
            ("preservation_root", self.preservation_root),
            ("archive_path", self.archive_path),
            ("compatibility_shell_path", self.compatibility_shell_path),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be non-empty text")
        if not _SHA256.fullmatch(self.archive_sha256):
            raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
        if not _GIT.fullmatch(self.jenny1_freeze_commit):
            raise ValueError("jenny1_freeze_commit must be a full lowercase Git commit")
        if not _GIT.fullmatch(self.jenny_next_head_commit):
            raise ValueError("jenny_next_head_commit must be a full lowercase Git commit")
        if self.runtime_import_status != "NOT_IMPORTED":
            raise ValueError("the genesis slice cannot import legacy runtime code")
        if self.personal_state_status != "NOT_IMPORTED":
            raise ValueError("the genesis slice cannot import personal state")
        if self.missions_status != "OMITTED" or self.consultants_status != "OMITTED":
            raise ValueError("missions and consultants must remain omitted")

    @property
    def genesis_ref(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self))

    @classmethod
    def owner_approved(cls, *, created_at_utc: str) -> "JennyGenesis":
        return cls(
            contract=GENESIS_CONTRACT,
            agent_ref=OWNER_AGENT_REF,
            created_at_utc=created_at_utc,
            preservation_root=OWNER_PRESERVATION_ROOT,
            archive_path=OWNER_ARCHIVE_PATH,
            archive_sha256=OWNER_ARCHIVE_SHA256,
            jenny1_freeze_commit=OWNER_JENNY1_FREEZE,
            jenny_next_head_commit=OWNER_JENNY_NEXT_HEAD,
            compatibility_shell_path=OWNER_COMPATIBILITY_SHELL,
        )

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> "JennyGenesis":
        if type(value) is not bytes:
            raise TypeError("genesis bytes must be exact bytes")
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("genesis bytes are not canonical JSON") from exc
        if type(payload) is not dict:
            raise ValueError("genesis payload must be an object")
        genesis = cls(**payload)
        if genesis.canonical_bytes() != value:
            raise ValueError("genesis bytes are not in canonical form")
        return genesis
