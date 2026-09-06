"""Canonical acquisitions and generic cognitive graph projections.

An acquisition assigns one already-canonical cognitive-memory record a local
Moving-Origin ordinal and binds it to one supported canonical source.  The
contract deliberately proves only this local binding.  Global uniqueness,
predecessor adjacency, persistence, and projection acknowledgement belong to
the future transaction-store integration.

Projection V2 is a disposable reference envelope.  It copies canonical
identity and content from an acquisition and conveys no truth, permission, or
authority of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, ClassVar, Mapping, Self

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    EpistemicStatus,
    _CanonicalContract,
    _digest,
    _exact_contract,
    _integer,
)
from angler.cognition.prospective_origin import (
    ProspectiveDynamicsBatch,
    ProspectiveResolution,
)


LEGACY_COGNITIVE_EPISODE_CONTRACT = (
    "ANG-CTR-COGNITIVE-EPISODE-001@0.1.0"
)
PROSPECTIVE_DYNAMICS_BATCH_CONTRACT = (
    "ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0"
)
PROSPECTIVE_RESOLUTION_CONTRACT = (
    "ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0"
)

SUPPORTED_ACQUISITION_SOURCE_CONTRACTS = frozenset(
    {
        LEGACY_COGNITIVE_EPISODE_CONTRACT,
        PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
        PROSPECTIVE_RESOLUTION_CONTRACT,
    }
)

_SOURCE_REF_ATTRIBUTES = {
    LEGACY_COGNITIVE_EPISODE_CONTRACT: "episode_ref",
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT: "batch_ref",
    PROSPECTIVE_RESOLUTION_CONTRACT: "resolution_ref",
}

_SOURCE_TYPES = {
    LEGACY_COGNITIVE_EPISODE_CONTRACT: CognitiveEpisode,
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT: ProspectiveDynamicsBatch,
    PROSPECTIVE_RESOLUTION_CONTRACT: ProspectiveResolution,
}

_SOURCE_RECORD_SEMANTICS = {
    LEGACY_COGNITIVE_EPISODE_CONTRACT: (
        CognitiveMemoryKind.EPISODIC,
        EpistemicStatus.OBSERVED,
    ),
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT: (
        CognitiveMemoryKind.COUNTERFACTUAL,
        EpistemicStatus.PROPOSED,
    ),
    PROSPECTIVE_RESOLUTION_CONTRACT: (
        CognitiveMemoryKind.EPISODIC,
        EpistemicStatus.OBSERVED,
    ),
}


def _source_contract(value: object) -> str:
    for contract, source_type in _SOURCE_TYPES.items():
        if type(value) is source_type:
            return contract
    raise TypeError("source must be an exact supported canonical contract type")


def _source_identity(value: object) -> tuple[str, str]:
    """Return a strict contract/ref pair for one canonical source object."""

    contract = _source_contract(value)
    canonical = getattr(value, "canonical_bytes", None)
    parser = getattr(_SOURCE_TYPES[contract], "from_json", None)
    if not callable(canonical) or not callable(parser):
        raise TypeError("source must provide canonical_bytes and from_json")
    payload = canonical()
    if type(payload) is not bytes or not payload:
        raise ValueError("source canonical payload must be non-empty exact bytes")
    try:
        restored = parser(payload)
    except Exception as exc:
        raise ValueError("source does not satisfy its canonical contract") from exc
    if restored != value or restored.canonical_bytes() != payload:
        raise ValueError("source does not round-trip through its canonical contract")

    attribute = _SOURCE_REF_ATTRIBUTES[contract]
    source_ref = getattr(value, attribute, None)
    _digest(source_ref, "source_ref")
    expected_ref = "sha256:" + hashlib.sha256(payload).hexdigest()
    if source_ref != expected_ref:
        raise ValueError("source reference does not identify its canonical bytes")
    return contract, source_ref


def _validate_source_contract(value: object) -> str:
    if type(value) is not str or value not in SUPPORTED_ACQUISITION_SOURCE_CONTRACTS:
        raise ValueError("source_contract is not supported for cognitive acquisition")
    return value


def _validate_record_binding(
    *,
    source_contract: str,
    source_ref: str,
    record: CognitiveMemoryRecord,
    ordinal: int | None = None,
) -> None:
    _validate_source_contract(source_contract)
    _digest(source_ref, "source_ref")
    if not isinstance(record, CognitiveMemoryRecord):
        raise TypeError("record must be a CognitiveMemoryRecord")
    # Re-run all record invariants at the acquisition/projection trust edge.
    encoded = record.canonical_bytes()
    try:
        restored = CognitiveMemoryRecord.from_json(encoded)
    except Exception as exc:
        raise ValueError("record does not satisfy its canonical contract") from exc
    if restored != record or restored.record_ref != record.record_ref:
        raise ValueError("record does not round-trip through its canonical contract")
    if ordinal is not None and record.acquired_ordinal != ordinal:
        raise ValueError("record acquisition ordinal does not match acquisition ordinal")
    if source_ref not in record.provenance_refs:
        raise ValueError("canonical source reference is absent from record provenance")
    expected_kind, expected_status = _SOURCE_RECORD_SEMANTICS[source_contract]
    if record.kind is not expected_kind or record.epistemic_status is not expected_status:
        raise ValueError("record kind or epistemic status does not match its source")


@dataclass(frozen=True, slots=True)
class CognitiveAcquisition(_CanonicalContract):
    """One locally ordered canonical record acquisition.

    This value does not claim that its ordinal is globally unique or that its
    predecessor is the database's actual head.  Those are transactional
    schema-v3 invariants, intentionally outside this standalone contract.
    """

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-ACQUISITION-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "acquisition_ref"

    ordinal: int
    predecessor_acquisition_ref: str | None
    source_contract: str
    source_ref: str
    record: CognitiveMemoryRecord

    def __post_init__(self) -> None:
        _integer(self.ordinal, "ordinal")
        _digest(
            self.predecessor_acquisition_ref,
            "predecessor_acquisition_ref",
            optional=True,
        )
        if (self.ordinal == 0) != (self.predecessor_acquisition_ref is None):
            raise ValueError(
                "ordinal zero must have no predecessor and every positive ordinal must have one"
            )
        _validate_record_binding(
            source_contract=self.source_contract,
            source_ref=self.source_ref,
            record=self.record,
            ordinal=self.ordinal,
        )

    @property
    def acquisition_ref(self) -> str:
        return self.content_ref

    @classmethod
    def from_source(
        cls,
        source: object,
        *,
        ordinal: int,
        predecessor_acquisition_ref: str | None,
        record: CognitiveMemoryRecord,
    ) -> Self:
        source_contract, source_ref = _source_identity(source)
        return cls(
            ordinal=ordinal,
            predecessor_acquisition_ref=predecessor_acquisition_ref,
            source_contract=source_contract,
            source_ref=source_ref,
            record=record,
        )

    def assert_source(self, source: object) -> None:
        source_contract, source_ref = _source_identity(source)
        if (
            source_contract != self.source_contract
            or source_ref != self.source_ref
        ):
            raise ValueError("source does not match the cognitive acquisition")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "ordinal": self.ordinal,
            "predecessor_acquisition_ref": self.predecessor_acquisition_ref,
            "record": self.record.to_payload(),
            "source_contract": self.source_contract,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "contract",
            "ordinal",
            "predecessor_acquisition_ref",
            "record",
            "source_contract",
            "source_ref",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        record_payload = payload["record"]
        if not isinstance(record_payload, Mapping):
            raise TypeError("acquisition record must be an object")
        return cls(
            ordinal=payload["ordinal"],
            predecessor_acquisition_ref=payload["predecessor_acquisition_ref"],
            source_contract=payload["source_contract"],
            source_ref=payload["source_ref"],
            record=CognitiveMemoryRecord.from_payload(record_payload),
        )


@dataclass(frozen=True, slots=True)
class CognitiveGraphProjectionV2(_CanonicalContract):
    """A rebuildable reference projection of one generic acquisition."""

    CONTRACT: ClassVar[str] = (
        "ANG-CTR-COGNITIVE-GRAPH-PROJECTION-002@0.1.0"
    )
    REF_ATTRIBUTE: ClassVar[str] = "projection_ref"

    acquisition_ref: str
    source_contract: str
    source_ref: str
    record: CognitiveMemoryRecord

    def __post_init__(self) -> None:
        _digest(self.acquisition_ref, "acquisition_ref")
        _validate_record_binding(
            source_contract=self.source_contract,
            source_ref=self.source_ref,
            record=self.record,
        )

    @property
    def projection_ref(self) -> str:
        return self.content_ref

    @classmethod
    def from_acquisition(cls, acquisition: CognitiveAcquisition) -> Self:
        if not isinstance(acquisition, CognitiveAcquisition):
            raise TypeError("acquisition must be a CognitiveAcquisition")
        return cls(
            acquisition_ref=acquisition.acquisition_ref,
            source_contract=acquisition.source_contract,
            source_ref=acquisition.source_ref,
            record=acquisition.record,
        )

    def assert_acquisition(self, acquisition: CognitiveAcquisition) -> None:
        if not isinstance(acquisition, CognitiveAcquisition):
            raise TypeError("acquisition must be a CognitiveAcquisition")
        if (
            self.acquisition_ref != acquisition.acquisition_ref
            or self.source_contract != acquisition.source_contract
            or self.source_ref != acquisition.source_ref
            or self.record != acquisition.record
        ):
            raise ValueError("projection does not match the canonical acquisition")

    def to_payload(self) -> dict[str, Any]:
        return {
            "acquisition_ref": self.acquisition_ref,
            "contract": self.CONTRACT,
            "record": self.record.to_payload(),
            "source_contract": self.source_contract,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "acquisition_ref",
            "contract",
            "record",
            "source_contract",
            "source_ref",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        record_payload = payload["record"]
        if not isinstance(record_payload, Mapping):
            raise TypeError("projection record must be an object")
        return cls(
            acquisition_ref=payload["acquisition_ref"],
            source_contract=payload["source_contract"],
            source_ref=payload["source_ref"],
            record=CognitiveMemoryRecord.from_payload(record_payload),
        )


__all__ = [
    "CognitiveAcquisition",
    "CognitiveGraphProjectionV2",
    "LEGACY_COGNITIVE_EPISODE_CONTRACT",
    "PROSPECTIVE_DYNAMICS_BATCH_CONTRACT",
    "PROSPECTIVE_RESOLUTION_CONTRACT",
    "SUPPORTED_ACQUISITION_SOURCE_CONTRACTS",
]
