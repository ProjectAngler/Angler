"""Networkless Cognee reference projection for Jenny 2.0 canonical episodes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence
from uuid import UUID, uuid5

from angler.memory.cognee_subprocess_bindings import CogneeSubprocessBindings
from angler.memory.cognee_worker_protocol import CogneeWorkerScope

from .higher_level_autonomy_adapter import CapabilityModuleHit, ReferenceMemoryHit
from .procedural_experience import procedural_projection_metadata
from .reading_experience import reading_projection_metadata


_NAMESPACE = UUID("8a3e6432-8e8e-5a43-9c61-60b1a1e58617")
_TARGET_NAMESPACE = UUID("87abf7f6-befd-5f28-88e9-da4453a4f568")
_CAPABILITY_NAMESPACE = UUID("db12feb4-d360-5c9d-84ee-c52bdfaa1558")
_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_CONTRACT = "ANG-CTR-PERSISTENT-AUTONOMY-STATE-001@0.1.0"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


class CogneeJennyReferenceBackend:
    """Cognee supplies candidate refs only; it never becomes a state writer."""

    def __init__(self, bindings: CogneeSubprocessBindings) -> None:
        self.bindings = bindings

    @classmethod
    async def start(
        cls,
        scope: CogneeWorkerScope,
        *,
        timeout_seconds: float = 120.0,
    ) -> "CogneeJennyReferenceBackend":
        bindings = await CogneeSubprocessBindings.start(
            scope=scope, timeout_seconds=timeout_seconds
        )
        return cls(bindings)

    async def project(
        self,
        *,
        record_ref: str,
        content: str,
        provenance_refs: Sequence[str],
        acquired_ordinal: int,
    ) -> str:
        if not _REF.fullmatch(record_ref):
            raise ValueError("record_ref must be a lowercase SHA-256 reference")
        refs = tuple(provenance_refs)
        if len(refs) != 2 or refs[0] != record_ref or not _REF.fullmatch(refs[1]):
            raise ValueError("projection provenance must be exact episode/event refs")
        if type(content) is not str or not content.strip() or len(content) > 16_384:
            raise ValueError("projection content must be bounded non-empty text")
        if type(acquired_ordinal) is not int or acquired_ordinal < 0:
            raise ValueError("acquired_ordinal must be non-negative")
        procedural = procedural_projection_metadata(content)
        typed = procedural if procedural is not None else reading_projection_metadata(content)
        memory_kind = "EPISODIC" if typed is None else typed.memory_kind
        epistemic_status = (
            "PROPOSED" if typed is None else typed.epistemic_status
        )
        adjacent_record_refs = (
            () if typed is None else typed.adjacent_episode_refs
        )
        acquisition_ref = _digest(
            {
                "record_ref": record_ref,
                "event_ref": refs[1],
                "ordinal": acquired_ordinal,
            }
        )
        projection_ref = _digest(
            {
                "acquisition_ref": acquisition_ref,
                "source_contract": _SOURCE_CONTRACT,
                "source_ref": refs[1],
                "record_ref": record_ref,
                "memory_kind": memory_kind,
                "epistemic_status": epistemic_status,
                "adjacent_record_refs": adjacent_record_refs,
            }
        )
        references = []
        for target_ref in adjacent_record_refs:
            target = self.bindings.DataPoint(
                id=uuid5(_TARGET_NAMESPACE, target_ref),
                record_ref=target_ref,
            )
            edge = self.bindings.Edge(
                relationship_type="FEEDBACK_ON",
                properties={"target_ref": target_ref},
            )
            references.append((edge, target))
        point = self.bindings.DataPoint(
            id=uuid5(_NAMESPACE, record_ref),
            acquisition_ref=acquisition_ref,
            source_contract=_SOURCE_CONTRACT,
            source_ref=refs[1],
            record_ref=record_ref,
            projection_ref=projection_ref,
            acquired_ordinal=acquired_ordinal,
            search_text=content,
            adjacent_record_refs=list(adjacent_record_refs),
            references=references,
            visibility="LEARNER_VISIBLE",
            memory_kind=memory_kind,
            epistemic_status=epistemic_status,
            belongs_to_set=[
                self.bindings.NodeSet(
                    id=uuid5(_NAMESPACE, self.bindings.node_set_name),
                    name=self.bindings.node_set_name,
                )
            ],
        )
        return await self.bindings.add_data_points([point])

    async def search(
        self, request: str, *, limit: int
    ) -> Sequence[ReferenceMemoryHit]:
        result = await self.bindings.search_references(request, limit=limit)
        entries = result[0]["search_result"]
        return tuple(
            ReferenceMemoryHit(
                record_ref=entry["payload"]["record_ref"],
                semantic_distance=float(entry["score"]),
            )
            for entry in entries
        )

    async def close(self) -> None:
        await self.bindings.close()


class CogneeJennyCapabilityBackend:
    """Separate procedural index returning only capability content hashes."""

    def __init__(self, bindings: CogneeSubprocessBindings) -> None:
        self.bindings = bindings

    @classmethod
    async def start(
        cls,
        scope: CogneeWorkerScope,
        *,
        timeout_seconds: float = 120.0,
    ) -> "CogneeJennyCapabilityBackend":
        bindings = await CogneeSubprocessBindings.start(
            scope=scope, timeout_seconds=timeout_seconds
        )
        return cls(bindings)

    async def project(
        self,
        *,
        record_ref: str,
        content: str,
        provenance_refs: Sequence[str],
        acquired_ordinal: int,
    ) -> str:
        if not _REF.fullmatch(record_ref):
            raise ValueError("record_ref must be a lowercase SHA-256 reference")
        refs = tuple(provenance_refs)
        if len(refs) != 2 or refs[0] != record_ref or not _REF.fullmatch(refs[1]):
            raise ValueError(
                "capability projection provenance must be capability and state refs"
            )
        if type(content) is not str or not content.strip() or len(content) > 16_384:
            raise ValueError("capability projection content must be bounded non-empty text")
        if type(acquired_ordinal) is not int or acquired_ordinal < 0:
            raise ValueError("acquired_ordinal must be non-negative")
        acquisition_ref = _digest(
            {
                "capability_ref": record_ref,
                "state_ref": refs[1],
                "ordinal": acquired_ordinal,
            }
        )
        projection_ref = _digest(
            {
                "acquisition_ref": acquisition_ref,
                "source_contract": _SOURCE_CONTRACT,
                "source_ref": refs[1],
                "record_ref": record_ref,
                "memory_kind": "PROCEDURAL",
            }
        )
        point = self.bindings.DataPoint(
            id=uuid5(_CAPABILITY_NAMESPACE, record_ref),
            acquisition_ref=acquisition_ref,
            source_contract=_SOURCE_CONTRACT,
            source_ref=refs[1],
            record_ref=record_ref,
            projection_ref=projection_ref,
            acquired_ordinal=acquired_ordinal,
            search_text=content,
            adjacent_record_refs=[],
            references=[],
            visibility="LEARNER_VISIBLE",
            memory_kind="PROCEDURAL",
            epistemic_status="MODEL_PROPOSAL_WITH_OBSERVED_CONSEQUENCE",
            belongs_to_set=[
                self.bindings.NodeSet(
                    id=uuid5(
                        _CAPABILITY_NAMESPACE, self.bindings.node_set_name
                    ),
                    name=self.bindings.node_set_name,
                )
            ],
        )
        return await self.bindings.add_data_points([point])

    async def search(
        self, request: str, *, limit: int
    ) -> Sequence[CapabilityModuleHit]:
        result = await self.bindings.search_references(request, limit=limit)
        entries = result[0]["search_result"]
        return tuple(
            CapabilityModuleHit(
                capability_ref=entry["payload"]["record_ref"],
                semantic_distance=float(entry["score"]),
            )
            for entry in entries
        )

    async def close(self) -> None:
        await self.bindings.close()
